//==================================================================================
// Rational-rescaling GPU rescale, unfused — see RationalRescale.cuh.
//==================================================================================
#include "CKKS/RationalRescale.cuh"

#include <algorithm>
#include <cassert>
#include <chrono>
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
        cudaMallocHost(&pin_stage, 2 * MAXP * sizeof(void*));
        cudaEventCreateWithFlags(&pin_evt, cudaEventDisableTiming);
    } else {
        cudaEventSynchronize(pin_evt);  // the previous upload must have drained before reuse
    }
    void** cpu_ptr    = pin_stage;
    void** cpu_auxptr = pin_stage + MAXP;
    for (size_t k = 0; k < n; ++k) {
        assert(limb[k].index() == U32 && "RR chains are all-U32 (< 2^30) by design");
        auto& l       = std::get<U32>(limb[k]);
        cpu_ptr[k]    = &l.v.data[0];
        cpu_auxptr[k] = &l.aux.data[0];
    }
    assert((int)n <= limbptr.size && (int)n <= MAXP);
    cudaMemcpyAsync(limbptr.data, cpu_ptr, n * sizeof(void*), cudaMemcpyHostToDevice, s.ptr());
    cudaMemcpyAsync(auxptr.data, cpu_auxptr, n * sizeof(void*), cudaMemcpyHostToDevice, s.ptr());
    cudaEventRecord(pin_evt, s.ptr());
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
    ws.adoptLimbPtrsFrom(src, cc.windowSize(level));

    RNSPoly& aux = cc.getKeySwitchAux2();
    out0.generateSpecialLimbs(false, false);
    out1.generateSpecialLimbs(false, false);
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
        out0.moddown(true, true, 0);
        out1.moddown(true, true, 1);
        tick(2, tmark);
        return;
    }

    // --- unfused reference: modup, then the standalone dot ---
    ws.modup(aux.GPU.at(0));

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
    out0.SetModUp(true);
    out1.SetModUp(true);
    out0.moddown(true, true, 0);
    out1.moddown(true, true, 1);
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

    // the textbook degree-2 product — the same three lines RRChain::EvalMultRelin computes
    RNSPoly c0(cc, level), c1(cc, level), c2(cc, level), tmp(cc, level);
    c0.multElement(A0, B0);
    c1.multElement(A0, B1);
    tmp.multElement(A1, B0);
    c1.add(tmp);
    c2.multElement(A1, B1);

    // relinearize: the degree-2 term is keyswitched and folded back in
    RNSPoly d0(cc, level), d1(cc, level);
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
    for (int it = -1; it < iters; ++it) {  // it == -1 warms
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        // RNSPoly has neither copy- nor move-ASSIGNMENT (reference members), so the running
        // ciphertext lives in an optional and each level move-CONSTRUCTS into it.
        std::optional<RNSPoly> c;
        c.emplace(cc, top_level);
        rrLoadWindow(cc, *c, a0, top_level);
        c->NTT(1, false);
        for (int r = top_level; r >= 1; --r) {
            // one circuit level: square, relinearize, rescale
            RNSPoly sq(cc, r), d0(cc, r), d1(cc, r), c2(cc, r);
            sq.multElement(*c, *c);
            c2.multElement(*c, *c);
            c2.SetModUp(false);
            RRKeySwitchCore(c2, cc.GetEvalKey(keyid), d0, d1);
            sq.add(d0);
            sq.rrRescale();        // sq is now at level r-1 and IS the next ciphertext
            c.emplace(std::move(sq));
        }
        cudaDeviceSynchronize();
        if (it >= 0)
            samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
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
