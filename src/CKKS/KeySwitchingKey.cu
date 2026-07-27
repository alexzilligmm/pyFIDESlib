//
// Created by carlosad on 26/09/24.
//

#include <algorithm>
#include <cstdlib>
#include <source_location>
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/RNSPoly.cuh"
#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
constexpr int PREFIX_SIZE = 0;
#else
#include <source_location>
using sc = std::source_location;
constexpr int PREFIX_SIZE = 23;
#endif

namespace FIDESlib::CKKS {

namespace {
/* Lever 1b-i (KSK bit-packing): uniform packed width for this chain's keys, or 0 = dense.
 * DEFAULT OFF — measured wall-NEUTRAL(+0.5 ms) on the n32 composite chain (gates 50427306 /
 * 50428528 after the compile-time-width + min-blocks-16 register fix, ncu 50428101): the
 * -12.5% KSK DRAM saving is eaten by the funnelshift unpack at ~94% warp occupancy. Bit-exact
 * ([bitcmp] 0 mismatches) with -12.5% key device memory, so FIDESLIB_KSK_PACK=1 remains the
 * memory-pressure opt-in; the packed plumbing is the substrate for Lever 1b-ii (seed-expanded
 * `a` — regeneration has no unpack tax). Eligible: single-GPU, all-u32 (type==0) chains with
 * max prime width in the instantiated set — the packed dot-kernel arms are u32-only, and n64
 * keys would need a uint2-straddling unpack that is not built. Lossless by construction. */
int kskPackBitsPolicy(Context& cc) {
    static const bool enabled = [] {
        const char* e = std::getenv("FIDESLIB_KSK_PACK");
        return e != nullptr && std::atoi(e) != 0;
    }();
    if (!enabled || cc->GPUid.size() != 1)
        return 0;
    const auto& hc = cc->precom.constants[0];
    if (hc.type != 0)
        return 0;
    const int K = (int)cc->splitSpecialMeta.at(0).size();
    int W = 0;
    for (int i = 0; i < cc->L + K; ++i)
        W = std::max<int>(W, (int)hc.prime_bits[i]);
    // Widths are compile-time in the dot kernels (register economy, ncu 50428101) — arm only
    // the instantiated set; other chains fall back dense (add an instantiation to extend).
    return (W == 27 || W == 28) ? W : 0;
}
}  // namespace

void KeySwitchingKey::Initialize(RawKeySwitchKey& rkk, int q_band) {
    CudaNvtxRange r(std::string{sc::current().function_name()}.substr());
    CKKS::SetCurrentContext(cc);
    keyID = rkk.keyid;

    if (q_band >= 0 && cc->GPUid.size() > 1)
        q_band = -1;  // banding is single-GPU only
    a.generateDecompAndDigit(true, q_band);
    b.generateDecompAndDigit(true, q_band);
    if (cc->GPUid.size() > 1) {
        a.grow(cc->L, false, true);
        b.grow(cc->L, false, true);
    }
    a.loadDecompDigit(rkk.r_key[0], rkk.r_key_moduli[0]);
    b.loadDecompDigit(rkk.r_key[1], rkk.r_key_moduli[1]);

    if (const int W = kskPackBitsPolicy(cc)) {
        a.GPU.at(0).packKeyLimbs(W);
        b.GPU.at(0).packKeyLimbs(W);
    }

    cudaDeviceSynchronize();
}

KeySwitchingKey::KeySwitchingKey(Context& cc)
    : my_range(loc, LIFETIME),
      keyID(""),
      cc((assert(cc != nullptr), CudaNvtxStart(std::string{sc::current().function_name()}.substr()), cc)),
      a(*cc, -1, false, true),
      b(*cc, -1, false, true) {
    CudaNvtxStop();
    /*
    if (cc.GPUid.size() > 1) {
        for (int j = 0; j < cc.dnum; ++j) {
            mgpu_a.emplace_back(cc, -1);
            mgpu_b.emplace_back(cc, -1);
        }
    }
     */
}
}  // namespace FIDESlib::CKKS
