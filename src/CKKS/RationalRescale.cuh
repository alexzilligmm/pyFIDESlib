//==================================================================================
// Rational-rescaling (Cheddar 25-30) GPU rescale — milestone (c).1, UNFUSED.
//
// One RR rescale step over a window of limbs (arbitrary global primeids, EVAL
// format, all on ONE stream): multiply every current limb by prod(added) —
// whose residue on each incoming prime is exactly 0, so the "add-back base
// extension" is a zero-extension — then divide out the dropped primes one at a
// time with the centered-lift exact division (INTT drop limb -> SwitchModulus
// lift -> NTT -> (x - lift) * d^{-1}). Operation-for-operation identical to the
// CPU reference lbcrypto::RRChain::RescaleElement (openfhe-1.4.2-rr.patch), so
// the two are bit-compatible.
//
// Correctness-first: sequential kernels, no fusion (the RESCALE2 pattern shows
// how to fuse later). U32 limbs only — every RR chain prime is < 2^31.
//==================================================================================
#ifndef FIDESLIB_CKKS_RATIONALRESCALE_CUH
#define FIDESLIB_CKKS_RATIONALRESCALE_CUH

#include <string>
#include <utility>
#include <vector>
#include "CKKS/Limb.cuh"

namespace FIDESlib::CKKS {

class ContextData;
class RNSPoly;

/**
 * Apply one RR rescale step in place.
 * @param limbs   the window's limbs, global-ascending primeid order, EVAL format,
 *                all Limb<uint32_t> on the same stream. Mutated: dropped limbs
 *                erased, added limbs (zero after the prod(added) scalar) inserted
 *                in primeid order.
 * @param drop    global primeids to divide out, processed IN THIS ORDER (must
 *                match the CPU reference order for bit-compat).
 * @param add     global primeids entering the window.
 */
void RRRescaleStep(ContextData& cc, std::vector<LimbImpl>& limbs, const std::vector<int>& drop,
                   const std::vector<int>& add, int gpuId = 0);

/**
 * Host-buffer harness around RRRescaleStep for out-of-library consumers (the
 * bit-compat gate): takes the window's limbs as COEFFICIENT-domain host vectors
 * plus their global primeids, runs load -> NTT -> RRRescaleStep -> INTT ->
 * store on the library side, and returns the next window's coefficient limbs.
 * ContextData carries #ifdef NCCL members, so its field offsets differ between
 * the fideslib build and consumers compiled without the define — all field
 * access must stay inside the library TU (see the note in Context.cuh).
 */
std::vector<std::vector<uint32_t>> RRRescaleStepHost(ContextData& cc,
                                                     const std::vector<std::vector<uint32_t>>& coeffLimbs,
                                                     const std::vector<int>& primeids, const std::vector<int>& drop,
                                                     const std::vector<int>& add);

class KeySwitchingKey;

/**
 * Milestone (c).3: HYBRID keyswitch of `c` at its RR level against the single evk stored at
 * P·Qmax, truncated to the level's window. Digits are `window ∩ global partition` and are
 * contiguous; the evk rows are read POSITIONALLY by global index, which is what makes a
 * single full-chain key serve every level with no per-level key material.
 *
 * `c` is consumed (modup'd in place, its special limbs left behind); `out0`/`out1` must
 * already sit at the same RR level and receive the (b, a) contributions ModDown'ed back to
 * the window basis — i.e. exactly RRChain::KeySwitchCore's return value.
 */
void RRKeySwitchCore(RNSPoly& c, const KeySwitchingKey& key, RNSPoly& out0, RNSPoly& out1);

/**
 * Milestone (c).2 harness: the same rescale step driven through the WINDOWED poly
 * representation (RNSPoly::grow/load/NTT/multScalar/rrRescale/INTT/store) instead of a bare
 * limb vector — i.e. through everything pbase touches. Takes the level's window as
 * coefficient-domain host limbs (ascending global primeid, i.e. slot order), optionally
 * multiplies by `scalar` in EVAL to exercise the batched elementwise path, rescales once,
 * and returns level-1's window. Same TU-boundary rule as RRRescaleStepHost.
 */
std::vector<std::vector<uint32_t>> RRPolyRescaleStepHost(ContextData& cc,
                                                         const std::vector<std::vector<uint32_t>>& coeffLimbs,
                                                         int level, uint64_t scalar = 1);

/**
 * Milestone (c).4b gate harness: a full RR ct*ct -> relinearize -> (optionally) rr-rescale,
 * i.e. RRChain::EvalMultRelin followed by RRChain::RescaleInPlace, driven end to end on the
 * GPU. Inputs are the two operands' components as coefficient-domain host limbs at RR level
 * `level`; the return is the output pair's windows, coefficient domain, at `level` (or at
 * `level - 1` when `rescale`). Same TU-boundary rule as the other harnesses here.
 */
std::pair<std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> RREvalMultRelinHost(
    ContextData& cc, const std::vector<std::vector<uint32_t>>& a0, const std::vector<std::vector<uint32_t>>& a1,
    const std::vector<std::vector<uint32_t>>& b0, const std::vector<std::vector<uint32_t>>& b1, int level,
    const std::string& keyid, bool rescale);

/**
 * Milestone (c).3 gate harness. Loads `c` (the degree-2 term) at RR level `level` from
 * coefficient-domain host limbs, keyswitches it against the context's eval key, and returns
 * the two output windows as coefficient-domain host limbs — {b-part, a-part}. Same
 * TU-boundary rule as the other harnesses here.
 */
std::pair<std::vector<std::vector<uint32_t>>, std::vector<std::vector<uint32_t>>> RRKeySwitchHost(
    ContextData& cc, const std::vector<std::vector<uint32_t>>& coeffLimbs, int level, const std::string& keyid);

// out-of-line field accessors for gate-side sanity checks
uint64_t RRPrimeAt(ContextData& cc, int primeid);
size_t RRNumPrimes(ContextData& cc);

// layout-mismatch canary for consumers compiled in a different TU environment
size_t RRDebugSizeofContextData();

}  // namespace FIDESlib::CKKS

#endif
