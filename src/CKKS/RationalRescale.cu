//==================================================================================
// Rational-rescaling GPU rescale, unfused — see RationalRescale.cuh.
//==================================================================================
#include "CKKS/RationalRescale.cuh"

#include <algorithm>
#include <cassert>
#include <optional>

#include "CKKS/Context.cuh"
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
                   const std::vector<int>& add) {
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
                Limb<uint32_t> nl(cc, 0, s, sortedAdd[ai]);
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
            Limb<uint32_t> scratch(cc, 0, s, t.primeid);
            SwitchModulus<uint32_t><<<N / blk, blk, 0, s.ptr()>>>(dl.v.data, d, scratch.v.data, t.primeid);
            scratch.NTT();
            t.sub(scratch);
            const auto dinv = static_cast<uint32_t>(modinv(dq % q, q));
            scalar_mult_<uint32_t, ALGO_BARRETT><<<N / blk, blk, 0, s.ptr()>>>(t.v.data, dinv, t.primeid);
        }
    }
}

}  // namespace FIDESlib::CKKS
