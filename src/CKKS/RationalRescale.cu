//==================================================================================
// Rational-rescaling GPU rescale, unfused — see RationalRescale.cuh.
//==================================================================================
#include "CKKS/RationalRescale.cuh"

#include <algorithm>
#include <cassert>
#include <optional>

#include "CKKS/Context.cuh"
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
                                                     const std::vector<int>& add) {
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
    RRRescaleStep(cc, limbs, drop, add);
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
    std::vector<void*> cpu_ptr(n, nullptr), cpu_auxptr(n, nullptr);
    for (size_t k = 0; k < n; ++k) {
        assert(limb[k].index() == U32 && "RR chains are all-U32 (< 2^30) by design");
        auto& l       = std::get<U32>(limb[k]);
        cpu_ptr[k]    = &l.v.data[0];
        cpu_auxptr[k] = &l.aux.data[0];
    }
    assert((int)n <= limbptr.size);
    cudaMemcpyAsync(limbptr.data, cpu_ptr.data(), n * sizeof(void*), cudaMemcpyHostToDevice, s.ptr());
    cudaMemcpyAsync(auxptr.data, cpu_auxptr.data(), n * sizeof(void*), cudaMemcpyHostToDevice, s.ptr());
    // The host staging vectors die at the end of this scope, so the uploads must have
    // consumed them by then (they are pageable, but do not lean on the driver's staging).
    cudaStreamSynchronize(s.ptr());
    CudaCheckErrorModNoSync;
}

void LimbPartition::rrRescale(const std::vector<int>& drop, const std::vector<int>& add, const int new_level) {
    cudaSetDevice(device);
    assert(cc.isRR() && "rrRescale is only defined on an RR chain");
    assert(cc.GPUid.size() == 1 && "RR is single-GPU until milestone (c).3");
    assert((int)limb.size() == cc.windowSize(new_level + 1));
    RRRescaleStep(cc, limb, drop, add, id);
    assert((int)limb.size() == cc.windowSize(new_level));
    pbase = cc.windowLo(new_level);
    assert(limb.empty() || PRIMEID(limb.front()) == pbase);
    // Both window edges moved and the limb vector was REBUILT (not appended to), so every
    // entry of the device pointer tables is stale — including slot 0's.
    refreshLimbPtrs();
}

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
// Host harness for the (c).2 gate. ContextData/RNSPoly both carry `#ifdef NCCL`
// members, so a consumer TU compiled without the define sees different field
// offsets and a different sizeof — the gate may not construct or touch either.
// It hands over plain coefficient-domain host limbs and gets them back.
//==================================================================================

std::vector<std::vector<uint32_t>> RRPolyRescaleStepHost(ContextData& cc,
                                                         const std::vector<std::vector<uint32_t>>& coeffLimbs,
                                                         const int level, const uint64_t scalar) {
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
    p.rrRescale();
    p.INTT(1, false);
    p.sync();

    std::vector<std::vector<uint64_t>> out64;
    p.store(out64);
    std::vector<std::vector<uint32_t>> out(out64.size());
    for (size_t k = 0; k < out64.size(); ++k)
        out[k].assign(out64[k].begin(), out64[k].end());
    return out;
}

}  // namespace FIDESlib::CKKS
