//==================================================================================
// Rational-rescaling GPU rescale, unfused — see RationalRescale.cuh.
//==================================================================================
#include "CKKS/RationalRescale.cuh"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <array>
#include <cstdlib>
#include <iostream>
#include <optional>

#include "CKKS/Context.cuh"
#include "CKKS/ElemenwiseBatchKernels.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/LimbPartition.cuh"
#include "CKKS/RNSPoly.cuh"
#include "LimbUtils.cuh"
#include "Math.cuh"
#include "ModMult.cuh"

namespace FIDESlib::CKKS {

/** TO-TRY §2.10b' probe sink: when non-null, RRKeySwitchCore accumulates its HOST ISSUE cost
 *  into [prep, specials, modup, dot, moddown]. Distinct from its `phase_ms`, which fences each
 *  phase with a device sync and so measures EXECUTION — the walk is issue-bound, so it is the
 *  issue split that decides what to cut. Single-threaded, like the rest of the RR path. */
double* rr_ks_host_ms = nullptr;
extern double* rr_resc_host_ms;

// defined (and instantiated) in Rescale.cu
template <typename T>
__global__ void SwitchModulus(const T* src, const int __grid_constant__ o_primeid, T* res,
                              const int __grid_constant__ n_primeid);

size_t RRDebugSizeofContextData() {
    return sizeof(ContextData);
}

uint64_t RRPrimeAt(ContextData& cc, int primeid) {
    return cc.prime.at(primeid).p;
}

size_t RRNumPrimes(ContextData& cc) {
    return cc.prime.size();
}

std::vector<std::vector<uint32_t>> RRRescaleStepHost(ContextData& cc,
                                                     const std::vector<std::vector<uint32_t>>& coeffLimbs,
                                                     const std::vector<int>& primeids, const std::vector<int>& drop,
                                                     const std::vector<int>& add, double* only_step_ms) {
    assert(coeffLimbs.size() == primeids.size());
    cudaSetDevice(cc.GPUid[0]);
    Stream s;
    s.init();
    std::vector<LimbImpl> limbs;
    limbs.reserve(coeffLimbs.size());
    for (size_t k = 0; k < coeffLimbs.size(); ++k) {
        Limb<uint32_t> l(cc, 0, s, primeids[k]);
        l.load(coeffLimbs[k]);
        l.NTT();
        limbs.emplace_back(std::move(l));
    }
    // Optional isolation for the (c).4 benchmark: time ONLY the step, not the harness's
    // per-call limb construction — which otherwise dominates and hides what is being compared.
    if (only_step_ms) {
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        RRRescaleStep(cc, limbs, drop, add);
        cudaDeviceSynchronize();
        *only_step_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    } else {
        RRRescaleStep(cc, limbs, drop, add);
    }
    std::vector<std::vector<uint32_t>> out(limbs.size());
    for (size_t k = 0; k < limbs.size(); ++k) {
        auto& l = std::get<U32>(limbs[k]);
        l.INTT();
        l.store(out[k]);
    }
    cudaStreamSynchronize(s.ptr());
    return out;
}

void RRRescaleStep(ContextData& cc, std::vector<LimbImpl>& limbs, const std::vector<int>& drop,
                   const std::vector<int>& add, const int gpuId) {
    assert(!limbs.empty());
    Stream& s     = STREAM(limbs[0]);
    const int N   = cc.N;
    const int blk = 128;

    // 1. t = c * prod(added) on every CURRENT limb (the dropped ones included —
    //    their centered lifts below must be lifts of t, exactly as on the CPU).
    if (!add.empty()) {
        for (auto& li : limbs) {
            assert(li.index() == U32 && "RR chains are all-U32 (< 2^31) by design");
            auto& l          = std::get<U32>(li);
            const uint64_t q = cc.prime[l.primeid].p;
            uint64_t acc     = 1;
            for (int a : add)
                acc = static_cast<uint64_t>(static_cast<unsigned __int128>(acc) * (cc.prime[a].p % q) % q);
            scalar_mult_<uint32_t, ALGO_BARRETT>
                <<<N / blk, blk, 0, s.ptr()>>>(l.v.data, static_cast<uint32_t>(acc), l.primeid);
        }
        // 2. the incoming primes' residues of t are exactly 0: zero-extension,
        //    merged keeping global-ascending primeid order. Limb has reference
        //    members (no assignment), so the vector is REBUILT via move-
        //    construction — never inserted into.
        std::vector<LimbImpl> merged;
        merged.reserve(limbs.size() + add.size());
        size_t li = 0, ai = 0;
        std::vector<int> sortedAdd = add;
        std::sort(sortedAdd.begin(), sortedAdd.end());
        while (li < limbs.size() || ai < sortedAdd.size()) {
            if (ai >= sortedAdd.size() || (li < limbs.size() && PRIMEID(limbs[li]) < sortedAdd[ai])) {
                merged.emplace_back(std::move(limbs[li++]));
            }
            else {
                Limb<uint32_t> nl(cc, gpuId, s, sortedAdd[ai]);
                cudaMemsetAsync(nl.v.data, 0, sizeof(uint32_t) * N, s.ptr());
                merged.emplace_back(std::move(nl));
                ++ai;
            }
        }
        limbs = std::move(merged);
    }

    // 3. divide out the dropped primes one at a time (centered-lift exact division)
    for (int d : drop) {
        std::vector<LimbImpl> kept;
        kept.reserve(limbs.size() - 1);
        std::optional<Limb<uint32_t>> dropped;
        for (auto& li : limbs) {
            if (PRIMEID(li) == d)
                dropped.emplace(std::move(std::get<U32>(li)));
            else
                kept.emplace_back(std::move(li));
        }
        assert(dropped.has_value());
        limbs             = std::move(kept);
        Limb<uint32_t>& dl = *dropped;

        dl.INTT();  // -> coefficient domain (result lands back in dl.v)
        const uint64_t dq = cc.prime[d].p;

        for (auto& ti : limbs) {
            auto& t          = std::get<U32>(ti);
            const uint64_t q = cc.prime[t.primeid].p;
            Limb<uint32_t> scratch(cc, gpuId, s, t.primeid);
            SwitchModulus<uint32_t><<<N / blk, blk, 0, s.ptr()>>>(dl.v.data, d, scratch.v.data, t.primeid);
            scratch.NTT();
            t.sub(scratch);
            const auto dinv = static_cast<uint32_t>(modinv(dq % q, q));
            scalar_mult_<uint32_t, ALGO_BARRETT><<<N / blk, blk, 0, s.ptr()>>>(t.v.data, dinv, t.primeid);
        }
    }
}

//==================================================================================
// Milestone (c).2 — the same step driven through the WINDOWED poly representation.
//==================================================================================

void LimbPartition::refreshLimbPtrs() {
    cudaSetDevice(device);
    const size_t n = limb.size();
    if (n == 0)
        return;
    // PINNED, partition-lifetime staging: the upload's source outlives the call, so no stream
    // sync is needed. It used to be a stack vector, which forced a cudaStreamSynchronize here
    // — twice per RR rescale, draining the pipeline each time.
    if (pin_stage == nullptr) {
        // TO-TRY §2.10b'.2: a RING, not one slot. With one slot the event below is recorded
        // after the whole rescale's kernels, so the "wait for the 512-byte copy" became a wait
        // for the GPU to drain a level — a per-level pipeline drain in an issue-bound path.
        // FIDESLIB_RR_PIN_RING=1 restores the old single-slot behaviour exactly (the ablation).
        static const int ring = [] {
            const char* e = std::getenv("FIDESLIB_RR_PIN_RING");
            const int v = e ? std::atoi(e) : 4;
            return v < 1 ? 1 : (v > PIN_RING_MAX ? PIN_RING_MAX : v);
        }();
        pin_ring = ring;
        cudaMallocHost(&pin_stage, (size_t)pin_ring * 2 * MAXP * sizeof(void*));
        for (int k = 0; k < pin_ring; ++k)
            cudaEventCreateWithFlags(&pin_evt[k], cudaEventDisableTiming);
    } else {
        // Only this SLOT's previous upload has to have drained, not the whole stream.
        cudaEventSynchronize(pin_evt[pin_slot]);
    }
    void** cpu_ptr    = pin_stage + (size_t)pin_slot * 2 * MAXP;
    void** cpu_auxptr = cpu_ptr + MAXP;
    for (size_t k = 0; k < n; ++k) {
        assert(limb[k].index() == U32 && "RR chains are all-U32 (< 2^30) by design");
        auto& l       = std::get<U32>(limb[k]);
        cpu_ptr[k]    = &l.v.data[0];
        cpu_auxptr[k] = &l.aux.data[0];
    }
    assert((int)n <= limbptr.size && (int)n <= MAXP);
    cudaMemcpyAsync(limbptr.data, cpu_ptr, n * sizeof(void*), cudaMemcpyHostToDevice, s.ptr());
    cudaMemcpyAsync(auxptr.data, cpu_auxptr, n * sizeof(void*), cudaMemcpyHostToDevice, s.ptr());
    cudaEventRecord(pin_evt[pin_slot], s.ptr());
    pin_slot = (pin_slot + 1) % pin_ring;
    CudaCheckErrorModNoSync;
}

// LimbPartition::rrRescale lives in LimbPartition.cu: it launches NTT_<..., NTT_RESCALE>,
// and a __global__ template launched from a TU that only sees its declaration gets a weak
// local stub with no device code (the 'invalid device function' trap, ElemenwiseBatchKernels.cuh).

void RNSPoly::rrRescale() {
    assert(cc.isRR() && "RNSPoly::rrRescale is only defined on an RR chain");
    if (level < 1)
        throw std::runtime_error("RR: cannot rescale below level 0");
    std::vector<int> drop, add;
    cc.rrRescaleSets(level, drop, add);
    const int new_level = level - 1;
    for (auto& g : GPU)
        g.rrRescale(drop, add, new_level);
    level = new_level;
}

//==================================================================================
// Milestone (c).3 — HYBRID keyswitch at an RR level.
//
// Unfused by design, exactly as (c).1's rescale was: modup -> dot -> moddown driven
// through the existing primitives, so the gate measures the WINDOW bookkeeping and
// nothing else. Fusing it into the shipping *ModupDotKSK paths is (c).4.
//==================================================================================

namespace {
/* Mirror of the (TU-local) kskRegenSeed/kskRegenEligible pair in LimbPartitionMGPU.cu — same
 * gates, same shape numbering. Duplicated rather than exported because that file is compiled
 * only in NCCL-enabled builds; keep the two in step if the gates ever change. */
const uint32_t* rrRegenSeed(const LimbPartition& ka, const int block_x, int* shape) {
    *shape          = 0;
    const int lvl   = kskRegenLevel();
    const bool elig = ka.ksk_seed_set && ka.cc.GPUid.size() == 1 && ka.cc.precom.constants[0].type == 0;
    if (lvl < 2 || !elig) {
        if (ka.ksk_a_released)
            throw std::runtime_error("RRKeySwitchCore: the key's `a` was released at load "
                                     "(FIDESLIB_KSK_REGEN>=2) but no regen arm is armed for this launch");
        return nullptr;
    }
    if (lvl >= 3) {
        *shape = 2;  // stage-A smem arm: diagnostic only, kept for arm parity
        return ka.ksk_seed;
    }
    if (ka.cc.N % (block_x * 16) != 0)  // stage B needs whole 16-coefficient threads
        return nullptr;
    *shape = 1;
    static const bool once = [] {  // run marker: proof the arm engaged, not just the env
        std::cerr << "[ksk_regen] active: RRKeySwitchCore regenerating kska from seed (stage B)\n";
        return true;
    }();
    (void)once;
    return ka.ksk_seed;
}
}  // namespace

void RRKeySwitchCore(RNSPoly& c, const KeySwitchingKey& key, RNSPoly& out0, RNSPoly& out1,
                     double* phase_ms) {
    // phase_ms, when given, is [modup, dot, moddown] in ms — each fenced by a device sync.
    // It exists to PRICE the remaining fusion before building it: the *ModupDotKSK path this
    // would fold into is also the classic shipping path's, so the prize has to justify the risk.
    auto tick = [&](int i, const std::chrono::steady_clock::time_point& t0) {
        if (!phase_ms) return std::chrono::steady_clock::now();
        cudaDeviceSynchronize();
        const auto t1 = std::chrono::steady_clock::now();
        phase_ms[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return t1;
    };
    if (phase_ms) cudaDeviceSynchronize();
    auto tmark = std::chrono::steady_clock::now();
    ContextData& cc = *key.cc;  // Context is a shared_ptr<ContextData>
    assert(cc.isRR() && "RRKeySwitchCore is only defined on an RR chain");
    assert(cc.GPUid.size() == 1 && "RR keyswitch is single-GPU");
    const int level = c.getLevel();
    if (out0.getLevel() != level || out1.getLevel() != level)
        throw std::runtime_error("RRKeySwitchCore: outputs must sit at the input's RR level");

    LimbPartition& src = c.GPU.at(0);
    LimbPartition& o0  = out0.GPU.at(0);
    LimbPartition& o1  = out1.GPU.at(0);
    const int id       = src.id;
    cudaSetDevice(src.device);

    /* FIDESLIB_RR_FUSED_KS (in-tree, DEFAULT 0 — MEASURED NEUTRAL, FAILURE §2.26).
     *
     * =1 folds modup's per-digit NTT into the KSK dot (NTT_KSK_DOT/_ACC), the shape
     * multModupDotKSK uses on a classic chain, so the extended digits never round-trip through
     * DRAM in eval form. Bit-identical to the default at all 9 gate windows.
     *
     * It measures NEUTRAL (3-pair alternating A/B, logN 16: -0.017 / -0.001 / -0.004 ms at
     * L26 / L13 / L5, signs split), and the traffic algebra says why it cannot do better:
     * fusing SAVES writing + re-reading the eval-domain digits, 2*dnum*(K+W) limb transfers,
     * but PAYS accumulator read-modify-writes, 4*dnum*(K+W), where the default arm's
     * fusedDotKSK_2_ accumulates across digits IN REGISTERS and writes each output row ONCE
     * (2*(K+W)). At dnum=4 that is 8 saved against 14 added — a net traffic INCREASE that only
     * comes out even because both working sets fit the 128 MB L2.
     *
     * So the default arm is not the unoptimized one: register accumulation across digits beats
     * per-digit DRAM accumulation, and this fusion is a step backwards that L2 happens to hide.
     * Kept in-tree because it is the shape the classic path uses, and because the balance flips
     * if dnum drops or the working set stops fitting L2. */
    static const bool fused_ks = [] {
        const char* e = std::getenv("FIDESLIB_RR_FUSED_KS");
        return e != nullptr && std::atoi(e) != 0;
    }();

    // The digits live in a CONTEXT-LIFETIME workspace, not on the ciphertext. It adopts the
    // input's limb pointers, so nothing is copied — it supplies only the DECOMP/DIGIT storage,
    // which is what a fresh ciphertext per level was otherwise re-allocating (~dnum*(K+L) limbs,
    // ~0.7 ms/level at logN 16 and scaling with dnum). See ContextData::rr_ks_workspace.
    RNSPoly& wsPoly = cc.getRRKeySwitchWorkspace();
    wsPoly.setLevel(level);
    LimbPartition& ws = wsPoly.GPU.at(0);
    // TO-TRY §2.10b': HOST ISSUE cost per sub-phase. Distinct from `phase_ms` above, which
    // fences each phase with a device sync and therefore measures EXECUTION. The walk is
    // issue-bound, so it is this breakdown that decides where to cut.
    auto hmark = std::chrono::steady_clock::now();
    auto hph = [&](int i) {
        const auto n = std::chrono::steady_clock::now();
        if (rr_ks_host_ms) rr_ks_host_ms[i] += std::chrono::duration<double, std::milli>(n - hmark).count();
        hmark = n;
    };
    ws.adoptLimbPtrsFrom(src, cc.windowSize(level));

    RNSPoly& aux = cc.getKeySwitchAux2();
    hph(0);  // prep: workspace adopt
    /* TO-TRY §2.10b': KEEP the outputs' special limbs alive across levels.
     *
     * MEASURED: allocating them was 94.3 us/level — 25.7 % of the walk's whole HOST ISSUE cost,
     * and the walk is issue-bound (quartering the ring moves it ~13 %). Each call was taking
     * ~12.6 MB per output through the pooled allocator (K specials x N x 2 x 8 B at dnum 4,
     * logN 16) and moddown handed it straight back, every level, forever.
     *
     * Safe because nothing here depends on their CONTENTS surviving or being fresh:
     * generateSpecialLimb is already idempotent (it no-ops unless SPECIALlimb is empty), it is
     * called with zero_out=false so the code ALREADY relies on the rows being fully written,
     * and the KSK dot writes every special row exactly once before moddown reads it. And
     * `free` does not control the modUp flag — RNSPoly::moddown ends with SetModUp(false)
     * either way — so only the storage persists.
     *
     * It pays because §2.10f made these outputs POOLED: a pooled d0/d1 is the same object at
     * the same level next time round, so "do not free" becomes "allocated once per level".
     * Ablation: FIDESLIB_RR_KEEP_SPECIALS=0. */
    static const bool keep_specials = [] {
        const char* e = std::getenv("FIDESLIB_RR_KEEP_SPECIALS");
        return e == nullptr || std::atoi(e) != 0;
    }();
    out0.generateSpecialLimbs(false, false);
    out1.generateSpecialLimbs(false, false);
    hph(1);  // special-limb allocation for the two outputs
    if (fused_ks) {
        static const bool once = [] {  // run marker: an A/B whose arms are the same arm is a lie
            std::cerr << "[rr_ks] active: FUSED modup+dot (FIDESLIB_RR_FUSED_KS=0 for the reference)\n";
            return true;
        }();
        (void)once;
        ws.rrModupDotKSK(o0, o1, key.a.GPU.at(0), key.b.GPU.at(0), aux.GPU.at(0));
        tmark = tick(0, tmark);      // the fused arm has no separate modup phase ...
        tmark = tick(1, tmark);      // ... so phase[0] is ~0 and phase[1] carries modup+dot
        out0.SetModUp(true);
        out1.SetModUp(true);
        out0.moddown(true, !keep_specials, 0);
        out1.moddown(true, !keep_specials, 1);
        tick(2, tmark);
        return;
    }

    // --- unfused reference: modup, then the standalone dot ---
    ws.modup(aux.GPU.at(0));
    hph(2);  // modup

    tmark = tick(0, tmark);

    // --- dot against the evk rows the window names ---
    const auto geoms   = cc.rrDigits(level);
    const int nSpecial = cc.rrNumSpecialInDigit();
    const int nLimbs   = cc.windowSize(level);
    const int dBase    = geoms.front().digit;
    const int nDigits  = (int)geoms.size();
    for (const auto& g : geoms)
        assert(g.digit == dBase + (&g - geoms.data()) && "active RR digits must be consecutive partitions");

    const LimbPartition& ka = key.a.GPU.at(0);
    const LimbPartition& kb = key.b.GPU.at(0);
    assert(ka.key_pack_bits == kb.key_pack_bits);

    Stream& s = src.getS();
    // modup ran on the WORKSPACE's stream, not the ciphertext's — before the workspace existed
    // these were the same stream and no fence was needed. Without this the dot can start on
    // half-built digits: it showed up as every window with pbase > 0 diverging while the
    // full-chain window (pbase 0) still passed.
    s.wait(ws.getS());
    s.wait(ka.getS());
    s.wait(kb.getS());
    s.wait(o0.getS());
    s.wait(o1.getS());

    // digits[] is indexed by GLOBAL partition, so inactive partitions simply stay null —
    // the kernel walks [dBase, dBase + nDigits).
    std::vector<void**> h_digits(cc.dnum * 6, nullptr);
    for (const auto& g : geoms) {
        const int d               = g.digit;
        h_digits[d]               = ws.DIGITlimbptr.at(d).data;
        h_digits[d + cc.dnum]     = ka.DIGITlimbptr.at(d).data;
        h_digits[d + 2 * cc.dnum] = kb.DIGITlimbptr.at(d).data;
        h_digits[d + 3 * cc.dnum] = src.limbptr.data;   // ciphertext limbs: SLOT-indexed
        h_digits[d + 4 * cc.dnum] = ka.limbptr.data;    // key limbs: full-chain, primeid-indexed
        h_digits[d + 5 * cc.dnum] = kb.limbptr.data;
    }
    if (cc.rr_digits_dev == nullptr) {
        cudaMalloc(&cc.rr_digits_dev, cc.dnum * 6 * sizeof(void**));
        cudaMallocHost(&cc.rr_digits_host, cc.dnum * 6 * sizeof(void**));
    }
    std::copy(h_digits.begin(), h_digits.end(), (void***)cc.rr_digits_host);
    void*** dDigits = cc.rr_digits_dev;
    cudaMemcpyAsync(dDigits, cc.rr_digits_host, cc.dnum * 6 * sizeof(void**), cudaMemcpyHostToDevice, s.ptr());
    // Argument order: the launcher's out1 collects the `a`-key product and out2 the `b`-key
    // product, whereas RRChain::KeySwitchCore returns (b-part, a-part) — so out1/out0 here.
    // Packed keys and in-kernel regen both work at RR levels (milestone (c).4a): packing is
    // per-row so it rides the same key-row index, and the regenerated `a` is a pure function
    // of (digit, prime) — the window only has to name the right prime. This is the shipping
    // config (FIDESLIB_KSK_PACK=1 FIDESLIB_KSK_REGEN=2), so RR must run it or no RR wall
    // number would mean anything.
    int regenShape                 = 0;
    const uint32_t* regenSeed      = rrRegenSeed(ka, 128, &regenShape);
    launchFusedDotKSK_2(dim3{(uint32_t)cc.N / 128, (uint32_t)(nSpecial + nLimbs)}, 128, s.ptr(), o1.limbptr.data,
                        o1.SPECIALlimbptr.data, o0.limbptr.data, o0.SPECIALlimbptr.data, dDigits, nDigits, id,
                        nSpecial, 0, ka.key_pack_bits, regenSeed, (uint32_t)cc.N >> 4, regenShape,
                        /*qbase=*/src.pbase, /*dbase=*/dBase);
    CudaCheckErrorModNoSync;  // pinned staging + a context-lifetime table: no sync, no per-call malloc

    o0.getS().wait(s);
    o1.getS().wait(s);
    ka.getS().wait(s);
    kb.getS().wait(s);
    ws.getS().wait(s);
    tmark = tick(1, tmark);

    // --- moddown back to the window basis ---
    hph(3);  // dot, including the digit-table build + upload
    out0.SetModUp(true);
    out1.SetModUp(true);
    out0.moddown(true, !keep_specials, 0);
    out1.moddown(true, !keep_specials, 1);
    hph(4);  // moddown, including the special-limb FREE
    tick(2, tmark);
}

//==================================================================================
// Host harness for the (c).2 gate. ContextData/RNSPoly both carry `#ifdef NCCL`
// members, so a consumer TU compiled without the define sees different field
// offsets and a different sizeof — the gate may not construct or touch either.
// It hands over plain coefficient-domain host limbs and gets them back.
//==================================================================================

std::vector<std::vector<uint32_t>> RRPolyRescaleStepHost(ContextData& cc,
                                                         const std::vector<std::vector<uint32_t>>& coeffLimbs,
                                                         const int level, const uint64_t scalar,
                                                         double* only_step_ms) {
    assert(cc.isRR());
    assert((int)coeffLimbs.size() == cc.windowSize(level));
    cudaSetDevice(cc.GPUid[0]);

    RNSPoly p(cc, level);
    const int lo = cc.windowLo(level);
    std::vector<std::vector<uint64_t>> data(coeffLimbs.size());
    std::vector<uint64_t> moduli(coeffLimbs.size());
    for (size_t k = 0; k < coeffLimbs.size(); ++k) {
        data[k].assign(coeffLimbs[k].begin(), coeffLimbs[k].end());
        moduli[k] = cc.prime.at(lo + k).p;
    }
    p.load(data, moduli);  // COEFFICIENT domain
    p.NTT(1, false);
    if (scalar != 1) {
        // Exercises the batched elementwise path (Scalar_mult_ over the window), which
        // resolves its primeids through C_.primeid_partition at base PB(0) = pbase — the
        // whole point of the window base. NOTE the scalar array is indexed by GLOBAL PRIMEID
        // (`b[primeid]` in the kernel), not by slot, so it stays full-chain-shaped.
        std::vector<uint64_t> elems(cc.prime.size(), 1);
        for (int k = 0; k < (int)coeffLimbs.size(); ++k)
            elems[lo + k] = scalar % cc.prime.at(lo + k).p;
        p.multScalar(elems);
    }
    if (only_step_ms) {
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        p.rrRescale();
        cudaDeviceSynchronize();
        *only_step_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    } else {
        p.rrRescale();
    }
    p.INTT(1, false);
    p.sync();

    std::vector<std::vector<uint64_t>> out64;
    p.store(out64);
    std::vector<std::vector<uint32_t>> out(out64.size());
    for (size_t k = 0; k < out64.size(); ++k)
        out[k].assign(out64[k].begin(), out64[k].end());
    return out;
}

namespace {

/* TO-TRY §2.10f: borrow level-keyed scratch from the context, or own it privately.
 *
 * The ablation is the point. `FIDESLIB_RR_SCRATCH_POOL=0` puts every borrower back to
 * constructing a fresh poly at every level — the shape the walk had when the per-level
 * construction measured 0.107 ms/level — so the pool's prize is an env flip, not a rebuild.
 *
 * Only scratch that KEEPS its level may be pooled; see ContextData::rr_scratch. */
bool rrScratchPooled() {
    static const bool pooled = [] {
        const char* e = std::getenv("FIDESLIB_RR_SCRATCH_POOL");
        return e == nullptr || std::atoi(e) != 0;
    }();
    return pooled;
}

class RRScratch {
    static constexpr int NSLOT = 8;
    ContextData& cc;
    const bool pooled;
    // FIXED storage, deliberately: callers hold REFERENCES across several get() calls, and a
    // std::vector that reallocates would move the polys out from under them (the pooled arm
    // is safe by construction — std::map references are stable). This bit at first build.
    std::array<std::optional<RNSPoly>, NSLOT> own;

   public:
    explicit RRScratch(ContextData& c) : cc(c), pooled(rrScratchPooled()) {}
    RNSPoly& get(const int level, const int slot) {
        assert(slot >= 0 && slot < NSLOT);
        if (pooled)
            return cc.getRRScratch(level, slot);
        own[slot].reset();             // destroy-then-construct, exactly the old scoped-local
        own[slot].emplace(cc, level);  // lifetime, so the OFF arm is the honest "before"
        return *own[slot];
    }
};

void rrLoadWindow(ContextData& cc, RNSPoly& p, const std::vector<std::vector<uint32_t>>& coeffLimbs, int level) {
    const int lo = cc.windowLo(level);
    std::vector<std::vector<uint64_t>> data(coeffLimbs.size());
    std::vector<uint64_t> moduli(coeffLimbs.size());
    for (size_t k = 0; k < coeffLimbs.size(); ++k) {
        data[k].assign(coeffLimbs[k].begin(), coeffLimbs[k].end());
        moduli[k] = cc.prime.at(lo + k).p;
    }
    p.load(data, moduli);
}

std::vector<std::vector<uint32_t>> rrStoreWindow(RNSPoly& p) {
    std::vector<std::vector<uint64_t>> out64;
    p.store(out64);
    std::vector<std::vector<uint32_t>> out(out64.size());
    for (size_t k = 0; k < out64.size(); ++k)
        out[k].assign(out64[k].begin(), out64[k].end());
    return out;
}
}  // namespace

std::pair<std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> RREvalMultRelinHost(
    ContextData& cc, const std::vector<std::vector<uint32_t>>& a0, const std::vector<std::vector<uint32_t>>& a1,
    const std::vector<std::vector<uint32_t>>& b0, const std::vector<std::vector<uint32_t>>& b1, const int level,
    const std::string& keyid, const bool rescale) {
    assert(cc.isRR());
    cudaSetDevice(cc.GPUid[0]);

    // load the four components of the two degree-1 operands, EVAL
    RNSPoly A0(cc, level), A1(cc, level), B0(cc, level), B1(cc, level);
    RNSPoly* in[4]                                     = {&A0, &A1, &B0, &B1};
    const std::vector<std::vector<uint32_t>>* src[4]   = {&a0, &a1, &b0, &b1};
    for (int k = 0; k < 4; ++k) {
        rrLoadWindow(cc, *in[k], *src[k], level);
        in[k]->NTT(1, false);
    }

    // the textbook degree-2 product — the same three lines RRChain::EvalMultRelin computes.
    // c0/c1 stay privately owned: the rescale arm moves them down a level, which is exactly
    // what pooled scratch may not do. tmp/c2/d0/d1 keep their level and are borrowed — which
    // is also what makes THIS test the pool's correctness gate: it is bit-exact against the
    // CPU reference over 7 windows x both rescale arms, and every window reuses the same
    // slots, so stale state from a previous borrow would show up here as a mismatch.
    RRScratch scratch(cc);
    RNSPoly c0(cc, level), c1(cc, level);
    RNSPoly& c2 = scratch.get(level, 2);
    RNSPoly& tmp = scratch.get(level, 3);
    c0.multElement(A0, B0);
    c1.multElement(A0, B1);
    tmp.multElement(A1, B0);
    c1.add(tmp);
    c2.multElement(A1, B1);

    // relinearize: the degree-2 term is keyswitched and folded back in
    RNSPoly& d0 = scratch.get(level, 0);
    RNSPoly& d1 = scratch.get(level, 1);
    RRKeySwitchCore(c2, cc.GetEvalKey(keyid), d0, d1);
    c0.add(d0);
    c1.add(d1);

    if (rescale) {
        c0.rrRescale();
        c1.rrRescale();
    }
    c0.INTT(1, false);
    c1.INTT(1, false);
    c0.sync();
    c1.sync();
    return {rrStoreWindow(c0), rrStoreWindow(c1)};
}

std::pair<std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> RRKeySwitchHost(
    ContextData& cc, const std::vector<std::vector<uint32_t>>& coeffLimbs, const int level, const std::string& keyid,
    double* phase_ms) {
    assert(cc.isRR());
    assert((int)coeffLimbs.size() == cc.windowSize(level));
    cudaSetDevice(cc.GPUid[0]);

    RNSPoly c(cc, level);
    rrLoadWindow(cc, c, coeffLimbs, level);
    c.NTT(1, false);

    RNSPoly out0(cc, level), out1(cc, level);
    RRKeySwitchCore(c, cc.GetEvalKey(keyid), out0, out1, phase_ms);

    out0.INTT(1, false);
    out1.INTT(1, false);
    out0.sync();
    out1.sync();
    return {rrStoreWindow(out0), rrStoreWindow(out1)};
}

double RREvalMultBenchHost(ContextData& cc, const std::vector<std::vector<uint32_t>>& a0,
                           const std::vector<std::vector<uint32_t>>& a1, const int level, const std::string& keyid,
                           const int iters) {
    assert(cc.isRR());
    cudaSetDevice(cc.GPUid[0]);
    RNSPoly A0(cc, level), A1(cc, level), c0(cc, level), c1(cc, level), c2(cc, level), tmp(cc, level);
    RNSPoly d0(cc, level), d1(cc, level);
    // ct*ct + relinearize, steady state; see the note at the end of the loop about the rescale.
    std::vector<double> samples;
    for (int it = -1; it < iters; ++it) {  // it == -1 warms and allocates
        rrLoadWindow(cc, A0, a0, level);
        rrLoadWindow(cc, A1, a1, level);
        A0.NTT(1, false);
        A1.NTT(1, false);
        c2.SetModUp(false);
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        // the payload-level op: three degree-2 products, one relinearization, one rescale
        c0.multElement(A0, A0);
        c1.multElement(A0, A1);
        tmp.multElement(A1, A0);
        c1.add(tmp);
        c2.multElement(A1, A1);
        RRKeySwitchCore(c2, cc.GetEvalKey(keyid), d0, d1);
        c0.add(d0);
        c1.add(d1);
        cudaDeviceSynchronize();
        if (it >= 0)
            samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        // The RESCALE is deliberately NOT in this loop. rrRescale moves c0/c1 down a level and
        // RNSPoly has no assignment operator, so putting them back would mean re-allocating
        // every iteration — and allocation scales with limbs and dnum, which is exactly the
        // axis under test. Time the rescale separately (RRPolyRescaleStepHost's only_step_ms)
        // and report the two columns.
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

double RRPayloadWalkHost(ContextData& cc, const std::vector<std::vector<uint32_t>>& a0, const int top_level,
                         const std::string& keyid, const int iters) {
    assert(cc.isRR());
    cudaSetDevice(cc.GPUid[0]);
    std::vector<double> samples;
    // TO-TRY §2.10f probe: HOST time spent constructing the four per-level scratch polys,
    // which is the slice the walk carries and test_classic_walk (ciphertexts built outside
    // its timer) does not. Reported as a median alongside the walk.
    std::vector<double> ctor_samples;
    RRScratch scratch(cc);
    std::array<double, 5> host_tot{};
    std::array<double, 5> ks_tot{};
    std::array<double, 5> rs_tot{};
    const bool host_probe = std::getenv("RR_WALK_HOST_PROBE") != nullptr;
    for (int it = -1; it < iters; ++it) {  // it == -1 warms
        double ctor_ms = 0;
        std::array<double, 5> host_ms{};
        std::array<double, 5> ks_ms{};
        std::array<double, 5> rs_ms{};
        rr_ks_host_ms = host_probe ? ks_ms.data() : nullptr;
        rr_resc_host_ms = host_probe ? rs_ms.data() : nullptr;
        // RNSPoly has neither copy- nor move-ASSIGNMENT (reference members), so the running
        // ciphertext lives in an optional and each level move-CONSTRUCTS into it.
        std::optional<RNSPoly> c;
        c.emplace(cc, top_level);
        rrLoadWindow(cc, *c, a0, top_level);
        c->NTT(1, false);
        // Load OUTSIDE the timer, so this measures the circuit's arithmetic and matches how
        // the classic walk is timed (its construction is an H2D of the whole ciphertext).
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = top_level; r >= 1; --r) {
            // one circuit level: square, relinearize, rescale
            const auto tc0 = std::chrono::steady_clock::now();
            // sq is the ONE poly that cannot be pooled: rrRescale moves it to r-1 and it
            // BECOMES the next ciphertext, so its level is not stable and it is not scratch.
            RNSPoly sq(cc, r);
            RNSPoly& d0 = scratch.get(r, 0);
            RNSPoly& d1 = scratch.get(r, 1);
            RNSPoly& c2 = scratch.get(r, 2);
            ctor_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc0).count();
            // TO-TRY §2.10b': HOST time per phase. The walk is host-bound — quartering the ring
            // (RR_SWEEP_LOGN) moves it only ~13 % — so what matters is which phase's ISSUE cost
            // dominates, not which kernel's execution does. Every phase here is async, so these
            // are pure issue costs; the device work lands in the trailing sync.
            auto ph = [&](int i, const std::chrono::steady_clock::time_point& t) {
                const auto n = std::chrono::steady_clock::now();
                host_ms[i] += std::chrono::duration<double, std::milli>(n - t).count();
                return n;
            };
            auto pm = std::chrono::steady_clock::now();
            sq.multElement(*c, *c);
            c2.multElement(*c, *c);
            pm = ph(0, pm);  // binomial multiply
            c2.SetModUp(false);
            RRKeySwitchCore(c2, cc.GetEvalKey(keyid), d0, d1);
            pm = ph(1, pm);  // keyswitch
            sq.add(d0);
            pm = ph(2, pm);  // add
            sq.rrRescale();        // sq is now at level r-1 and IS the next ciphertext
            pm = ph(3, pm);  // rescale
            c.emplace(std::move(sq));
            ph(4, pm);       // hand-off
        }
        cudaDeviceSynchronize();
        if (it >= 0) {
            samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            ctor_samples.push_back(ctor_ms);
            for (int i = 0; i < 5; ++i) {
                host_tot[i] += host_ms[i];
                ks_tot[i] += ks_ms[i];
                rs_tot[i] += rs_ms[i];
            }
        }
    }
    rr_ks_host_ms = nullptr;
    rr_resc_host_ms = nullptr;
    if (std::getenv("RR_WALK_HOST_PROBE")) {
        const char* nm[5] = {"binomial mult", "keyswitch", "add", "rescale", "hand-off"};
        double sum = 0;
        for (int i = 0; i < 5; ++i)
            sum += host_tot[i] / iters;
        fprintf(stderr, "[rr_walk_host] ISSUE cost per walk (host, mean over %d runs), total %6.3f ms:\n", iters, sum);
        for (int i = 0; i < 5; ++i)
            fprintf(stderr, "[rr_walk_host]   %-14s %7.3f ms  -> %6.1f us/level  (%4.1f%%)\n", nm[i],
                    host_tot[i] / iters, 1000.0 * host_tot[i] / iters / top_level, 100.0 * host_tot[i] / iters / sum);
        const char* kn[5] = {"ks:prep", "ks:specials", "ks:modup", "ks:dot", "ks:moddown"};
        for (int i = 0; i < 5; ++i)
            fprintf(stderr, "[rr_walk_host]     %-12s %7.3f ms  -> %6.1f us/level  (%4.1f%% of walk issue)\n", kn[i],
                    ks_tot[i] / iters, 1000.0 * ks_tot[i] / iters / top_level, 100.0 * ks_tot[i] / iters / sum);
        const char* rn[5] = {"rs:multScalar", "rs:merge", "rs:refresh1", "rs:droploop", "rs:refresh2"};
        for (int i = 0; i < 5; ++i)
            fprintf(stderr, "[rr_walk_host]     %-12s %7.3f ms  -> %6.1f us/level  (%4.1f%% of walk issue)\n", rn[i],
                    rs_tot[i] / iters, 1000.0 * rs_tot[i] / iters / top_level, 100.0 * rs_tot[i] / iters / sum);
    }
    if (std::getenv("RR_WALK_CTOR_PROBE")) {
        std::sort(ctor_samples.begin(), ctor_samples.end());
        fprintf(stderr, "[rr_walk_ctor] per-level scratch construction (host, median over the run): %7.3f ms"
                        "  -> %6.3f ms/level\n",
                ctor_samples[ctor_samples.size() / 2], ctor_samples[ctor_samples.size() / 2] / top_level);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void RRKeySwitchBenchHost(ContextData& cc, const std::vector<std::vector<uint32_t>>& coeffLimbs, const int level,
                          const std::string& keyid, const int iters, double* phase_ms) {
    assert(cc.isRR());
    cudaSetDevice(cc.GPUid[0]);
    RNSPoly c(cc, level), out0(cc, level), out1(cc, level);
    // MEDIAN, not mean: this box is shared, and a single contention spike (measured: one L5
    // dot reading of 0.604 ms among 0.169/0.178) moves a 5-sample mean by 3x. Same lesson the
    // wall A/B harness records — high iteration count PLUS a robust statistic.
    std::vector<std::vector<double>> samples(3);
    for (int it = -1; it < iters; ++it) {  // it == -1 is the warm/allocating pass
        rrLoadWindow(cc, c, coeffLimbs, level);
        c.NTT(1, false);
        c.SetModUp(false);
        double ph[3] = {0, 0, 0};
        RRKeySwitchCore(c, cc.GetEvalKey(keyid), out0, out1, ph);
        if (it >= 0)
            for (int i = 0; i < 3; ++i)
                samples[i].push_back(ph[i]);
    }
    for (int i = 0; i < 3; ++i) {
        std::sort(samples[i].begin(), samples[i].end());
        phase_ms[i] = samples[i][samples[i].size() / 2];
    }
}


}  // namespace FIDESlib::CKKS
