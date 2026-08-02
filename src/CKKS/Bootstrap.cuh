//
// Created by carlosad on 4/12/24.
//

#ifndef GPUCKKS_BOOTSTRAP_CUH
#define GPUCKKS_BOOTSTRAP_CUH

#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "forwardDefs.cuh"
#include "pke/openfhe.h"

namespace FIDESlib::CKKS {
/// Stage-divergence harness hook (default nullptr = off): install a vector to receive a
/// clone of the ciphertext at each bootstrap stage checkpoint (pre-CtS, post-CtS, pre-StC,
/// post-StC, end). Single-threaded use only; caller must reset to nullptr afterwards.
// Per-stage checkpoints. Holds the HOST-side RawCipherText (plus the level, which the raw form
// does not carry), NOT a GPU ciphertext clone: cloning a windowed RR ciphertext is unsupported --
// Ciphertext(cc)+copy reaches the source level through prefix moves that an RR WINDOW has no
// meaning for -- whereas `store()` is already RR-aware and the GPU->OpenFHE conversion it feeds is
// gated by the probe's [rr_roundtrip] check. Same reason the CPU oracle reads through store().
struct BtsStageCheckpoint {
    std::string stage;
    int level;
    RawCipherText raw;
};
extern std::vector<BtsStageCheckpoint>* g_btsStageStash;
void BootstrapCPUraise(
    Ciphertext& ctxt, const int slots,
    std::shared_ptr<
        lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<expdtype>>>>>& CPUcc,
    lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys, const bool prescaled);
// void Bootstrap(Ciphertext& ctxt, const int slots, const bool prescaled = false);
void Bootstrap(Ciphertext& ctxt, const int slots, const bool prescaled = false);
double GetPreScaleFactor(Context& cc, int slots);
void ModRaise(Ciphertext& ctxt, const int slots, const int32_t correction, const bool prescaled = false,
              bool sparse_encaps = false);
}  // namespace FIDESlib::CKKS

#endif  //GPUCKKS_BOOTSTRAP_CUH
