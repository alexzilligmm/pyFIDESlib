//
// Created by carlosad on 4/12/24.
//

#include "CKKS/AccumulateBroadcast.cuh"
#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Bootstrap.cuh"
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/CoeffsToSlots.cuh"
#include "CKKS/Context.cuh"
#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
#else
#include <source_location>
using sc = std::source_location;
#endif

using namespace FIDESlib::CKKS;

constexpr bool PRINT = false;

// Pair with FIDESLIB_DA_FOLD (ApproxModEval.cu): the 2^correction recovery is
// folded into the last double-angle iteration, so the end-of-bootstrap integer
// scale-back must be skipped.
// Stage-divergence harness (default off): when a caller installs a stash vector, every
// btsStageProbe checkpoint (pre-CtS / post-CtS / pre-StC / post-StC / end) also deposits the
// HOST-side RawCipherText the caller can wrap and decrypt offline. Zero cost when null.
std::vector<FIDESlib::CKKS::BtsStageCheckpoint>* FIDESlib::CKKS::g_btsStageStash = nullptr;


// Effective correction factor for this bootstrap call: the ContextData override (armed by
// the wrapper's CorrectionScope) wins over the per-slots precomputation value.
static uint32_t effCorrectionFactor(FIDESlib::CKKS::ContextData& cc, int slots) {
    return cc.correctionFactorOverride >= 0
               ? (uint32_t)cc.correctionFactorOverride
               : cc.GetBootPrecomputation(slots).correctionFactor;
}

// n32 speed: the composite-port debug barriers. Each BTS_DIAG site used to be an
// UNCONDITIONAL cudaDeviceSynchronize() + cudaGetLastError() in the bootstrap hot path;
// three of them drained the whole device per bootstrap. Now off unless FIDESLIB_BTS_DIAG is
// set, and the env is read once (a getenv per call is itself measurable in a tight loop).
static bool btsDiagOn() {
    static const bool v = [] {
        const char* e = std::getenv("FIDESLIB_BTS_DIAG");
        return e && *e && *e != '0';
    }();
    return v;
}
#define BTS_DIAG(what)                                                                              \
    do {                                                                                            \
        if (btsDiagOn()) {                                                                          \
            cudaDeviceSynchronize();                                                                \
            cudaError_t _e = cudaGetLastError();                                                    \
            if (_e != cudaSuccess)                                                                  \
                std::cerr << "[diag] " what " ERROR: " << cudaGetErrorString(_e) << " at "          \
                          << __FILE__ << ":" << __LINE__ << std::endl;                              \
        }                                                                                           \
    } while (0)

static bool btsSfDebugOn() {
    static const bool v = [] { return std::getenv("BTS_SF_DEBUG") != nullptr; }();
    return v;
}

static void btsStageProbe(const char* stage, FIDESlib::CKKS::Ciphertext& ctxt) {
    if (FIDESlib::CKKS::g_btsStageStash) {
        cudaDeviceSynchronize();
        // Store to the HOST rather than cloning on the device: `store()` handles the RR window
        // (slot k -> global primeid windowLo+k) and the conversion that consumes it is verified
        // by the probe's round-trip check. A device clone is not available for RR -- see the
        // struct comment in Bootstrap.cuh.
        FIDESlib::CKKS::BtsStageCheckpoint cp;
        cp.stage = stage;
        cp.level = ctxt.getLevel();
        ctxt.store(cp.raw);
        FIDESlib::CKKS::g_btsStageStash->emplace_back(std::move(cp));
    }
    if (!btsSfDebugOn())
        return;
    cudaDeviceSynchronize();
    printf("[bts_stage] %s: level=%d deg=%d log2(NF)=%.3f table=%.3f\n", stage, ctxt.getLevel(), ctxt.NoiseLevel,
           std::log2(ctxt.NoiseFactor), std::log2(ctxt.cc.param.ScalingFactorReal[ctxt.getLevel()]));
}

static bool skipCorFactor() {
    static const bool v = [] {
        const char* e = std::getenv("FIDESLIB_SKIP_CORFACTOR");
        return e && *e && *e != '0';
    }();
    return v;
}

// RATIONAL RESCALING — the four structural "come down to the bottom before raising" sites.
//
// On a classic chain `dropToLevel(d-1)` is exact and FREE: discard the top limbs and the
// ciphertext is at the bottom modulus, same scale. An RR chain has no such move. Level r-1's
// window CONTAINS primes level r's does not (the smalls entering at the low edge), so there
// is nothing to discard your way to, and no intermediate stop either — holding `lo` while
// dropping top mains yields a prime set that is not any level of the schedule. The ONLY way
// down is the rescale, and it moves exactly one level.
//
// So the RR translation of "drop to the bottom" is not a drop at all: it is a PRECONDITION.
// The ciphertext must already have arrived at RR level 0 by rescaling, which the chain's own
// schedule does — the bootstrap is entered at level 1 and the adjust's rescale is the last
// step down. This throws with the level it actually found rather than silently rescaling a
// ciphertext the caller did not intend to move: a ciphertext that is not at the bottom here
// is a FLOW error (someone bootstrapped with payload levels left), and quietly consuming
// them would hide it while destroying precision.
static void rrRequireBottom(FIDESlib::CKKS::Ciphertext& ctxt, const char* site) {
    if (ctxt.getLevel() != 0)
        throw std::runtime_error(std::string("RR bootstrap: ") + site + " expects the ciphertext at RR level 0 (" +
                                 "the bottom window), found level " + std::to_string(ctxt.getLevel()) +
                                 ". An RR chain has no free level drop — the ciphertext must ARRIVE at the bottom "
                                 "by rescaling. Enter the bootstrap at level 1 (non-prescaled) or 0 (prescaled).");
}

void FIDESlib::CKKS::BootstrapCPUraise(
    Ciphertext& ctxt, const int slots,
    std::shared_ptr<
        lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<expdtype>>>>>& CPUcc,
    lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys, const bool prescaled) {
    CudaNvtxRange r(std::string{sc::current().function_name()});

    FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
    ContextData& cc = ctxt.cc;
    Ciphertext aux(cc_);
    bool isLT = cc.GetBootPrecomputation(slots).LT.slots == slots;

    /////////////////////////////////////////////////////////////////////
    //NativeInteger q = elementParamsRaisedPtr->GetParams()[0]->GetModulus().ConvertToInt();
    // COMPOSITESCALING: level 0 spans compositeDegree primes — the bootstrap's q0 is their
    // PRODUCT (~2^54 on a 2x27-bit chain). RATIONAL RESCALING: level 0 is a WINDOW of the
    // layout, not a prefix, so its product is taken over [windowLo(0), windowHi(0)]
    // (~2^78 on our schedule). Everything downstream (deg, correction, pre/post, the ModRaise
    // CRT lift) is derived from it — see ContextData::bottomModulus.
    const double qDouble = cc.bottomModulus();

    if constexpr (PRINT) {
        std::cout << "q0(bottom): ";
        std::cout << qDouble << std::endl;
    }
    const auto p = cc.param.raw->p;  //cryptoParams->GetPlaintextModulus();
    // RR: `p` is GetPlaintextModulus() = the scalingModSize (25), which is the size of a SMALL
    // PRIME, not the scale. The ciphertext carries sf(top) ~ 2^55 into the raise, so deg read 53
    // where the honest figure is 23 -- and deg drives correction, pre/post and the whole scale
    // bookkeeping. Mirrors the CPU fix in ckksrns-fhe.cpp; see docs/RR_BTS_RUNLOG.md.
    double powP = cc.isRR() ? cc.sfAtTop() : pow(2, p);

    if constexpr (PRINT) {
        std::cout << "p: " << p << std::endl;
    }
    int32_t deg = std::round(std::log2(qDouble / powP));
    // Guard restored (was commented out upstream): deg = q0_bits - scale_bits
    // must not exceed the correction factor (OpenFHE auto = 9), or the uint32
    // subtraction below underflows and corFactor = 1 << garbage poisons every
    // bootstrap SILENTLY (cost us a 6-config param sweep of tok0 garbage).
    // RATIONAL RESCALING: deg is structurally LARGE here and the guard's premise inverts.
    // The RR bottom is Cheddar's L0 = {q0, 2 tau} — a THREE-limb ~2^78 window — while the
    // scale the ciphertext carries into the raise is sf(top) ~ 2^55, so q0/sf ~ 2^23 against
    // the classic chain's 2^2. The correction factor exists to make the message SMALLER
    // relative to q0 ("emulate a larger q0"); an RR chain needs the opposite sign, which
    // `uint32_t correction` and the `1 << correction` recovery cannot express.
    //
    // So on RR the correction is set to ZERO rather than to a negative number: no extra
    // down-scaling, self-consistent scale bookkeeping (the mixed-chain arm below normalizes
    // by sf/q0 and the StC plaintexts carry the q0/sf[0] recovery), and the ~log2(q0/sf)
    // bits of headroom simply go unused. That is a PRECISION cost, not a correctness one,
    // and it is the honest way to expose it: measure it at Phase 6 rather than hide it.
    // The fix is a chain-design one — level 0's window is ~20 bits larger than the scale it
    // has to hold — see docs/RR_BTS_RUNLOG.md, Phase 3.
    // RR: the correction is SIGNED, and disabling it (the previous arm here) buried the message.
    // `correction = correctionFactor - deg` counts the bits the message is scaled DOWN before the
    // raise; an RR chain has the opposite problem -- its bottom window dwarfs the scale -- so it
    // must be scaled UP, i.e. a NEGATIVE correction. ModRaise already applies pow(2,-correction)
    // as a double and handles either sign; what could not was `uint32_t` and the `1 << correction`
    // recovery, both fixed here and at the recovery site. CPU twin: ckksrns-fhe.cpp `rrSigned`.
    const bool rrSigned = cc.isRR() && deg > static_cast<int32_t>(effCorrectionFactor(cc, slots));
    if (rrSigned) {
        // The bound is on what actually lands in the bottom modulus: the ciphertext enters at
        // scale powP and is scaled up by 2^-correction, so |m| * powP * 2^-correction < q0/(2K).
        const double maxUp = std::log2(qDouble / (powP * 2.0 * cc.GetBootK()));
        const int32_t up   = deg - (int32_t)effCorrectionFactor(cc, slots);
        if ((double)up > maxUp)
            throw std::runtime_error("RR: the signed correction asks to scale the message up by 2^" +
                                     std::to_string(up) + ", but the sine's input range allows only 2^" +
                                     std::to_string(maxUp));
        static bool announced = false;
        if (!announced) {
            announced = true;
            std::fprintf(stderr,
                         "[rr_bts] signed correction: deg=%d correctionFactor=%u -> correction=%d "
                         "(message scaled UP by 2^%d, bound 2^%.2f)\n",
                         deg, effCorrectionFactor(cc, slots), -up, up, maxUp);
        }
    } else if (deg > static_cast<int32_t>(effCorrectionFactor(cc, slots))) {
        throw std::runtime_error(
            "Bootstrap: deg=log2(q0/2^p)=" + std::to_string(deg) +
            " exceeds correctionFactor=" +
            std::to_string(effCorrectionFactor(cc, slots)) +
            " (uint32 underflow); pick q0_bits - scale_bits <= correctionFactor.");
    }
    int32_t correction = static_cast<int32_t>(effCorrectionFactor(cc, slots)) - deg;
    if constexpr (PRINT)
        std::cout << effCorrectionFactor(cc, slots) << " " << deg << std::endl;
    double post = std::pow(2, static_cast<double>(deg));

    double pre = 1. / post;
    uint64_t scalar = std::llround(post);

    // Mixed-size chain (see OpenFHE ckksrns-fhe.cpp, same gate): the uniform
    // identity sf[0] ~ 2^p * 2^deg does not hold; follow the COMPOSITESCALING
    // constants: pre = sf[0]/q0 input normalization, no integer 2^deg recovery
    // (the CPU-precomputed StC matrices carry scaleDec = q0/sf[0]).
    bool mixedChain = std::fabs(std::log2(cc.sfAtTop() * post / qDouble)) > 0.5;
    // COMPOSITESCALING always uses the sf/q0 normalization (OpenFHE: pre = sf[0]/qDouble,
    // no integer 2^deg recovery) — the same constants the mixed-chain arm implements.
    // RR is included unconditionally: with the corrected deg the mixedChain test reads ~0, and
    // falling through would restore the integer 2^deg recovery that the StC plaintexts already
    // carry as scaleDec = q0/sf[0] -- the double-count the CPU measured as a clean 2^deg jump.
    if (mixedChain || cc.compositeDegree() > 1 || cc.isRR()) {
        pre    = cc.sfAtTop() / qDouble;
        scalar = 1;
    }

    //////////////////////////////////////////////////////////////////////

    {

        ModRaise(ctxt, slots, correction, prescaled);

        //------------------------------------------------------------------------------
        // SETTING PARAMETERS FOR APPROXIMATE MODULAR REDUCTION
        //------------------------------------------------------------------------------

        // Coefficients of the Chebyshev series interpolating 1/(2 Pi) Sin(2 Pi K x)
        double k = cc.GetBootK();

        double constantEvalMult = pre * (1.0 / (k * cc.N));

        if constexpr (PRINT)
            std::cout << "mult: " << constantEvalMult << std::endl;
        ctxt.multScalar(constantEvalMult, false);

        if constexpr (PRINT) {
            std::cout << "Raise scaled ";
            for (auto& j : ctxt.c0.GPU) {
                cudaSetDevice(j.device);
                for (auto& i : j.limb) {
                    SWITCH(i, printThisLimb(1));
                }
            }
            std::cout << std::endl;
        }

        ////////////////////////////////////////////////////////////////

        Accumulate(ctxt, cc.GetBootPrecomputation(slots).accumulate_bStep, slots, cc.N / 2 / slots);
    }

    if (ctxt.NoiseLevel == 2) {
        ctxt.rescale();
    }

    btsStageProbe("pre-CtS", ctxt);
    if (isLT) {
        EvalLinearTransform(ctxt, slots, false);
    } else {
        EvalCoeffsToSlots(ctxt, slots, false);
    }
    btsStageProbe("post-CtS", ctxt);
    //  std::cout << "ModRed" << std::endl;

    if (cc.N / 2 == slots) {
        aux.conjugate(ctxt);
        Ciphertext ctxtEncI(cc_);
        ctxtEncI.sub(ctxt, aux);
        ctxt.add(aux);
        ctxtEncI.multMonomial(3 * 2 * cc.N / 4);
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxt.rescale();
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxtEncI.rescale();

        approxModReduction(ctxt, ctxtEncI, cc.GetEvalKey(ctxt.keyID), scalar);
    } else {
        aux.conjugate(ctxt);
        ctxt.add(aux);
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxt.rescale();
        approxModReductionSparse(ctxt, scalar);
    }

    if (ctxt.NoiseLevel == 2) {
        ctxt.rescale();
    }

    // FIDESLIB_CORFACTOR_PRE=1: apply the correction restore BEFORE SlotsToCoeffs.
    // StC is linear (StC(2^c x) = 2^c StC(x)) so the final value is identical, but the
    // stage-budget harness (2026-07-26) showed StC injects a CONSTANT ~8e-5 (2^-13.5)
    // absolute error while its EvalMod input is ~2^-25 clean: amplifying the SIGNAL
    // through StC instead of amplifying StC's noise afterwards removes the 2^correction
    // error blow-up entirely (the q0-headroom reason for the small message dies at
    // EvalMod). Default off = byte-identical legacy behaviour.
    static const bool corPre = [] {
        const char* e = std::getenv("FIDESLIB_CORFACTOR_PRE");
        return e && *e && *e != '0';
    }();
    // Undoing a NEGATIVE correction is a DIVISION, which `1 << correction` cannot express (and
    // shifting by a negative count is UB). Dividing the message by 2^k is identical to declaring
    // the scaling factor 2^k larger -- exact and noiseless, where an integer multiply would add
    // noise. Applied after StC below. CPU twin: the rrSigned arm of ckksrns-fhe.cpp.
    const bool rrSignedRecovery = cc.isRR() && correction < 0;
    uint64_t corFactor = rrSignedRecovery ? 1u : ((uint64_t)1 << std::llround(correction));
    if (corPre && !skipCorFactor() && corFactor != 1)
        multIntScalar(ctxt, corFactor);

    btsStageProbe("pre-StC", ctxt);
    if (isLT) {
        EvalLinearTransform(ctxt, slots, true);
    } else {
        EvalCoeffsToSlots(ctxt, slots, true);
    }
    btsStageProbe("post-StC", ctxt);

    if (cc.N / 2 != slots) {
        aux.rotate(ctxt, slots);
        ctxt.add(aux);
    }

    if (!corPre && !skipCorFactor() && corFactor != 1)
        multIntScalar(ctxt, corFactor);
    static const bool rrNoRecovery = [] {
        const char* e = std::getenv("RR_BTS_NORECOVERY");
        return e && *e && *e != '0';
    }();
    if (rrSignedRecovery && !skipCorFactor() && !rrNoRecovery) {
        if (std::getenv("RR_BTS_SCALEDBG"))
            std::fprintf(stderr, "[rr_scale] recovery: NoiseFactor 2^%.2f -> 2^%.2f (x2^%d)\n",
                         std::log2(ctxt.NoiseFactor), std::log2(ctxt.NoiseFactor) - (double)correction,
                         (int)-correction);
        ctxt.NoiseFactor *= std::pow(2.0, -(double)correction);
    }
    // Mixed-size chain: realize the pending StC rescale so the output lands
    // deg-1 exactly on the per-level table at the data scale (the lazy deg-2
    // state does not match ScalingFactorRealBig there). Uniform unchanged.
    // RR must be included: with the corrected deg the mixedChain test reads ~0, but StC's LAST
    // stage still sits on the bootstrap->payload boundary and carries the gated scale
    // sf[r-1]*F(r)/sf[r], which only lands on sf[r-1] AFTER a rescale. The bootstrap ends there,
    // so without this the output is left deg-2 at a scale matching no level -- measured as
    // 2^94 against the level table's 2^40. CPU twin: the tail of ckksrns-fhe.cpp's rrSigned arm.
    if ((mixedChain || cc.isRR()) && ctxt.NoiseLevel == 2)
        ctxt.rescale();
    btsStageProbe("end", ctxt);
    if constexpr (PRINT) {
        cudaDeviceSynchronize();
        std::cout << "End bootstrap ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(2));
            }
        }
        std::cout << std::endl;
        cudaDeviceSynchronize();
    }
}

void FIDESlib::CKKS::Bootstrap(Ciphertext& ctxt, const int slots, const bool prescaled) {
    CudaNvtxRange r(std::string{sc::current().function_name()});

    assert(slots >= ctxt.slots);
    int old_slots = ctxt.slots;

    FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
    ContextData& cc = ctxt.cc;
    Ciphertext aux(cc_);
    bool isLT = cc.GetBootPrecomputation(slots).LT.slots == slots;

    /////////////////////////////////////////////////////////////////////
    //NativeInteger q = elementParamsRaisedPtr->GetParams()[0]->GetModulus().ConvertToInt();
    // COMPOSITESCALING: level 0 spans compositeDegree primes — the bootstrap's q0 is their
    // PRODUCT (~2^54 on a 2x27-bit chain). RATIONAL RESCALING: level 0 is a WINDOW of the
    // layout, not a prefix, so its product is taken over [windowLo(0), windowHi(0)]
    // (~2^78 on our schedule). Everything downstream (deg, correction, pre/post, the ModRaise
    // CRT lift) is derived from it — see ContextData::bottomModulus.
    const double qDouble = cc.bottomModulus();

    if constexpr (PRINT) {
        std::cout << "q0(bottom): ";
        std::cout << qDouble << std::endl;
    }
    const auto p = cc.param.raw->p;  //cryptoParams->GetPlaintextModulus();
    // RR: `p` is GetPlaintextModulus() = the scalingModSize (25), which is the size of a SMALL
    // PRIME, not the scale. The ciphertext carries sf(top) ~ 2^55 into the raise, so deg read 53
    // where the honest figure is 23 -- and deg drives correction, pre/post and the whole scale
    // bookkeeping. Mirrors the CPU fix in ckksrns-fhe.cpp; see docs/RR_BTS_RUNLOG.md.
    double powP = cc.isRR() ? cc.sfAtTop() : pow(2, p);

    if constexpr (PRINT) {
        std::cout << "p: " << p << std::endl;
    }
    int32_t deg = std::round(std::log2(qDouble / powP));
    // Guard restored (was commented out upstream): deg = q0_bits - scale_bits
    // must not exceed the correction factor (OpenFHE auto = 9), or the uint32
    // subtraction below underflows and corFactor = 1 << garbage poisons every
    // bootstrap SILENTLY (cost us a 6-config param sweep of tok0 garbage).
    // RATIONAL RESCALING: deg is structurally LARGE here and the guard's premise inverts.
    // The RR bottom is Cheddar's L0 = {q0, 2 tau} — a THREE-limb ~2^78 window — while the
    // scale the ciphertext carries into the raise is sf(top) ~ 2^55, so q0/sf ~ 2^23 against
    // the classic chain's 2^2. The correction factor exists to make the message SMALLER
    // relative to q0 ("emulate a larger q0"); an RR chain needs the opposite sign, which
    // `uint32_t correction` and the `1 << correction` recovery cannot express.
    //
    // So on RR the correction is set to ZERO rather than to a negative number: no extra
    // down-scaling, self-consistent scale bookkeeping (the mixed-chain arm below normalizes
    // by sf/q0 and the StC plaintexts carry the q0/sf[0] recovery), and the ~log2(q0/sf)
    // bits of headroom simply go unused. That is a PRECISION cost, not a correctness one,
    // and it is the honest way to expose it: measure it at Phase 6 rather than hide it.
    // The fix is a chain-design one — level 0's window is ~20 bits larger than the scale it
    // has to hold — see docs/RR_BTS_RUNLOG.md, Phase 3.
    // RR: the correction is SIGNED, and disabling it (the previous arm here) buried the message.
    // `correction = correctionFactor - deg` counts the bits the message is scaled DOWN before the
    // raise; an RR chain has the opposite problem -- its bottom window dwarfs the scale -- so it
    // must be scaled UP, i.e. a NEGATIVE correction. ModRaise already applies pow(2,-correction)
    // as a double and handles either sign; what could not was `uint32_t` and the `1 << correction`
    // recovery, both fixed here and at the recovery site. CPU twin: ckksrns-fhe.cpp `rrSigned`.
    const bool rrSigned = cc.isRR() && deg > static_cast<int32_t>(effCorrectionFactor(cc, slots));
    if (rrSigned) {
        // The bound is on what actually lands in the bottom modulus: the ciphertext enters at
        // scale powP and is scaled up by 2^-correction, so |m| * powP * 2^-correction < q0/(2K).
        const double maxUp = std::log2(qDouble / (powP * 2.0 * cc.GetBootK()));
        const int32_t up   = deg - (int32_t)effCorrectionFactor(cc, slots);
        if ((double)up > maxUp)
            throw std::runtime_error("RR: the signed correction asks to scale the message up by 2^" +
                                     std::to_string(up) + ", but the sine's input range allows only 2^" +
                                     std::to_string(maxUp));
        static bool announced = false;
        if (!announced) {
            announced = true;
            std::fprintf(stderr,
                         "[rr_bts] signed correction: deg=%d correctionFactor=%u -> correction=%d "
                         "(message scaled UP by 2^%d, bound 2^%.2f)\n",
                         deg, effCorrectionFactor(cc, slots), -up, up, maxUp);
        }
    } else if (deg > static_cast<int32_t>(effCorrectionFactor(cc, slots))) {
        throw std::runtime_error(
            "Bootstrap: deg=log2(q0/2^p)=" + std::to_string(deg) +
            " exceeds correctionFactor=" +
            std::to_string(effCorrectionFactor(cc, slots)) +
            " (uint32 underflow); pick q0_bits - scale_bits <= correctionFactor.");
    }
    int32_t correction = static_cast<int32_t>(effCorrectionFactor(cc, slots)) - deg;
    if (std::getenv("BTS_SF_DEBUG"))
        fprintf(stderr, "[bts_cf] cc=%p override=%d deg=%d correction=%u\n", (void*)&cc,
                cc.correctionFactorOverride, deg, correction);
    if constexpr (PRINT)
        std::cout << effCorrectionFactor(cc, slots) << " " << deg << std::endl;
    double post = std::pow(2, static_cast<double>(deg));

    double pre = 1. / post;
    uint64_t scalar = std::llround(post);

    // Mixed-size chain (see OpenFHE ckksrns-fhe.cpp, same gate): the uniform
    // identity sf[0] ~ 2^p * 2^deg does not hold; follow the COMPOSITESCALING
    // constants: pre = sf[0]/q0 input normalization, no integer 2^deg recovery
    // (the CPU-precomputed StC matrices carry scaleDec = q0/sf[0]).
    bool mixedChain = std::fabs(std::log2(cc.sfAtTop() * post / qDouble)) > 0.5;
    // COMPOSITESCALING always uses the sf/q0 normalization (OpenFHE: pre = sf[0]/qDouble,
    // no integer 2^deg recovery) — the same constants the mixed-chain arm implements.
    // RR is included unconditionally: with the corrected deg the mixedChain test reads ~0, and
    // falling through would restore the integer 2^deg recovery that the StC plaintexts already
    // carry as scaleDec = q0/sf[0] -- the double-count the CPU measured as a clean 2^deg jump.
    if (mixedChain || cc.compositeDegree() > 1 || cc.isRR()) {
        pre    = cc.sfAtTop() / qDouble;
        scalar = 1;
    }

    //////////////////////////////////////////////////////////////////////
    bool sparse_encaps = cc.GetBootPrecomputation(slots).sparse_encaps;

    {
        ModRaise(ctxt, slots, correction, prescaled, sparse_encaps);
        //------------------------------------------------------------------------------
        // SETTING PARAMETERS FOR APPROXIMATE MODULAR REDUCTION
        //------------------------------------------------------------------------------

        // Coefficients of the Chebyshev series interpolating 1/(2 Pi) Sin(2 Pi K x)
        double k = cc.GetBootK();

        // TO-DO: The 1/32 scale will be pre-applied with OpenFHE v1.4, so remove it from here
        double constantEvalMult = pre * (1.0 / (k * cc.N));

        /*
        if (sparse_encaps) {
            constantEvalMult = pre * (1.0 / (k * cc.N) / 32);
        }
        */

        if constexpr (PRINT)
            std::cout << "mult: " << constantEvalMult << std::endl;
        ctxt.multScalar(constantEvalMult, false);

        // n32 speed: this was an UNCONDITIONAL cudaDeviceSynchronize() diagnostic left over from
        // the composite port -- a full device drain in the bootstrap hot path, right before the
        // heaviest stages, killing cross-stage stream overlap. Now gated (FIDESLIB_BTS_DIAG=1).
        BTS_DIAG("AFTER multScalar(constantEvalMult)");
        if constexpr (PRINT) {
            std::cout << "Raise scaled ";
            for (auto& j : ctxt.c0.GPU) {
                cudaSetDevice(j.device);
                for (auto& i : j.limb) {
                    SWITCH(i, printThisLimb(1));
                }
            }
            std::cout << std::endl;
        }

        ////////////////////////////////////////////////////////////////

        Accumulate(ctxt, cc.GetBootPrecomputation(slots).accumulate_bStep, slots, cc.N / 2 / slots);
    }

    ctxt.slots = cc.N / 2 == slots ? slots : 2 * slots;

    if (ctxt.NoiseLevel == 2) {
        ctxt.rescale();
        BTS_DIAG("AFTER rescale(NoiseLevel==2)");
    }

    btsStageProbe("pre-CtS", ctxt);
    if (isLT) {
        EvalLinearTransform(ctxt, slots, false);
    } else {
        EvalCoeffsToSlots(ctxt, slots, false);
    }
    btsStageProbe("post-CtS", ctxt);
    //  std::cout << "ModRed" << std::endl;

    if (cc.N / 2 == slots) {
        aux.conjugate(ctxt);
        Ciphertext ctxtEncI(cc_);
        ctxtEncI.sub(ctxt, aux);
        ctxt.add(aux);
        ctxtEncI.multMonomial(3 * 2 * cc.N / 4);
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxt.rescale();
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxtEncI.rescale();
        approxModReduction(ctxt, ctxtEncI, cc.GetEvalKey(ctxt.keyID), scalar);
    } else {
        aux.conjugate(ctxt);
        ctxt.add(aux);
        if (cc.rescaleTechnique == CKKS::FIXEDMANUAL)
            ctxt.rescale();
        approxModReductionSparse(ctxt, scalar);
    }

    if (ctxt.NoiseLevel == 2) {
        ctxt.rescale();
    }

    // FIDESLIB_CORFACTOR_PRE=1: apply the correction restore BEFORE SlotsToCoeffs.
    // StC is linear (StC(2^c x) = 2^c StC(x)) so the final value is identical, but the
    // stage-budget harness (2026-07-26) showed StC injects a CONSTANT ~8e-5 (2^-13.5)
    // absolute error while its EvalMod input is ~2^-25 clean: amplifying the SIGNAL
    // through StC instead of amplifying StC's noise afterwards removes the 2^correction
    // error blow-up entirely (the q0-headroom reason for the small message dies at
    // EvalMod). Default off = byte-identical legacy behaviour.
    static const bool corPre = [] {
        const char* e = std::getenv("FIDESLIB_CORFACTOR_PRE");
        return e && *e && *e != '0';
    }();
    // Undoing a NEGATIVE correction is a DIVISION, which `1 << correction` cannot express (and
    // shifting by a negative count is UB). Dividing the message by 2^k is identical to declaring
    // the scaling factor 2^k larger -- exact and noiseless, where an integer multiply would add
    // noise. Applied after StC below. CPU twin: the rrSigned arm of ckksrns-fhe.cpp.
    const bool rrSignedRecovery = cc.isRR() && correction < 0;
    uint64_t corFactor = rrSignedRecovery ? 1u : ((uint64_t)1 << std::llround(correction));
    if (corPre && !skipCorFactor() && corFactor != 1)
        multIntScalar(ctxt, corFactor);

    btsStageProbe("pre-StC", ctxt);
    if (isLT) {
        EvalLinearTransform(ctxt, slots, true);
    } else {
        EvalCoeffsToSlots(ctxt, slots, true);
    }
    btsStageProbe("post-StC", ctxt);

    if (cc.N / 2 != slots) {
        aux.rotate(ctxt, slots);
        ctxt.add(aux);
    }

    if (!corPre && !skipCorFactor() && corFactor != 1)
        multIntScalar(ctxt, corFactor);
    static const bool rrNoRecovery = [] {
        const char* e = std::getenv("RR_BTS_NORECOVERY");
        return e && *e && *e != '0';
    }();
    if (rrSignedRecovery && !skipCorFactor() && !rrNoRecovery) {
        if (std::getenv("RR_BTS_SCALEDBG"))
            std::fprintf(stderr, "[rr_scale] recovery: NoiseFactor 2^%.2f -> 2^%.2f (x2^%d)\n",
                         std::log2(ctxt.NoiseFactor), std::log2(ctxt.NoiseFactor) - (double)correction,
                         (int)-correction);
        ctxt.NoiseFactor *= std::pow(2.0, -(double)correction);
    }
    // Mixed-size chain: realize the pending StC rescale so the output lands
    // deg-1 exactly on the per-level table at the data scale (the lazy deg-2
    // state does not match ScalingFactorRealBig there). Uniform unchanged.
    // RR must be included: with the corrected deg the mixedChain test reads ~0, but StC's LAST
    // stage still sits on the bootstrap->payload boundary and carries the gated scale
    // sf[r-1]*F(r)/sf[r], which only lands on sf[r-1] AFTER a rescale. The bootstrap ends there,
    // so without this the output is left deg-2 at a scale matching no level -- measured as
    // 2^94 against the level table's 2^40. CPU twin: the tail of ckksrns-fhe.cpp's rrSigned arm.
    if ((mixedChain || cc.isRR()) && ctxt.NoiseLevel == 2)
        ctxt.rescale();
    btsStageProbe("end", ctxt);
    if constexpr (PRINT) {
        cudaDeviceSynchronize();
        std::cout << "End bootstrap ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(2));
            }
        }
        std::cout << std::endl;
        cudaDeviceSynchronize();
    }

    ctxt.slots = old_slots;
}

double FIDESlib::CKKS::GetPreScaleFactor(Context& cc_, int slots) {
    ContextData& cc = *cc_;
    SetCurrentContext(cc_);
    /////////////////////////////////////////////////////////////////////
    //NativeInteger q = elementParamsRaisedPtr->GetParams()[0]->GetModulus().ConvertToInt();
    // COMPOSITESCALING: level 0 spans compositeDegree primes — the bootstrap's q0 is their
    // PRODUCT (~2^54 on a 2x27-bit chain). RATIONAL RESCALING: level 0 is a WINDOW of the
    // layout, not a prefix, so its product is taken over [windowLo(0), windowHi(0)]
    // (~2^78 on our schedule). Everything downstream (deg, correction, pre/post, the ModRaise
    // CRT lift) is derived from it — see ContextData::bottomModulus.
    const double qDouble = cc.bottomModulus();

    if constexpr (PRINT) {
        std::cout << "q0(bottom): ";
        std::cout << qDouble << std::endl;
    }
    const auto p = cc.param.raw->p;  //cryptoParams->GetPlaintextModulus();
    // RR: `p` is GetPlaintextModulus() = the scalingModSize (25), which is the size of a SMALL
    // PRIME, not the scale. The ciphertext carries sf(top) ~ 2^55 into the raise, so deg read 53
    // where the honest figure is 23 -- and deg drives correction, pre/post and the whole scale
    // bookkeeping. Mirrors the CPU fix in ckksrns-fhe.cpp; see docs/RR_BTS_RUNLOG.md.
    double powP = cc.isRR() ? cc.sfAtTop() : pow(2, p);

    if constexpr (PRINT) {
        std::cout << "p: " << p << std::endl;
    }
    int32_t deg = std::round(std::log2(qDouble / powP));
    /*
    #if NATIVEINT != 128
        if (deg > static_cast<int32_t>(m_correctionFactor)) {
            OPENFHE_THROW("Degree [" + std::to_string(deg) + "] must be less than or equal to the correction factor [" +
                          std::to_string(m_correctionFactor) + "].");
        }
    #endif
        */
    // Same RR arm as the two bootstrap variants above: on an RR chain deg exceeds the
    // correction factor structurally (the bottom window is ~2^78 against a ~2^55 scale), and
    // the correction is set to zero rather than to an inexpressible negative.
    uint32_t correction = (cc.isRR() && deg > static_cast<int32_t>(effCorrectionFactor(cc, slots)))
                              ? 0u
                              : effCorrectionFactor(cc, slots) - deg;

    double res = 0.0;
    if (cc.rescaleTechnique == CKKS::FLEXIBLEAUTO || cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT) {
        const int d_ = cc.compositeDegree();
        uint32_t lvl = cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT;
        // RATIONAL RESCALING: the same three quantities keyed by LEVEL instead of by limb. The
        // pre-raise ciphertext sits at RR level 1 (one rescale above the bottom), the adjust's
        // rescale takes it to level 0, and that rescale divides the scale by
        // F(1) = prod(dropped)/prod(added) — a RATIO, since an RR rescale also ADDS primes.
        // The classic arms must not be EVALUATED on an RR chain: sfAtLimb now throws there.
        // composite: the pre-raise ciphertext sits at 2 LEVELS = 2d limbs; its scale lives at
        // limb 2d-1 and the adjust's rescale drops the top d primes (their product).
        const bool rr_     = cc.isRR();
        double targetSF    = rr_ ? cc.sfAtLevel(cc.topLevel()) : cc.sfAtLimb(cc.L - lvl * d_);
        double sourceSF    = rr_ ? cc.sfAtLevel(1) : cc.sfAtLimb(2 * d_ - 1);
        uint32_t numTowers = rr_ ? (uint32_t)cc.windowSize(1) : (uint32_t)(2 * d_);
        double modToDrop   = rr_ ? cc.rrRescaleFactor(1) : cc.modReduceProduct(2 * d_ - 1);
        //cryptoParams->GetElementParams()->GetParams()[numTowers - 1]->GetModulus().ConvertToDouble();
        // in the case of FLEXIBLEAUTO, we need to bring the ciphertext to the right scale using a
        // a scaling multiplication. Note the at currently FLEXIBLEAUTO is only supported for NATIVEINT = 64.
        // So the other branch is for future purposes (in case we decide to add add the FLEXIBLEAUTO support
        // for NATIVEINT = 128.
        // Scaling down the message by a correction factor to emulate using a larger q0.
        // This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        double adjustmentFactor = (targetSF / sourceSF) * (modToDrop / sourceSF);
        double pow = std::pow((double)2.0, (double)-1.0 * (double)correction);
        adjustmentFactor *= pow;
        if (cc.isRR() && std::getenv("RR_BTS_SCALEDBG"))
            std::fprintf(stderr, "[rr_scale] adjust: correction=%d pow=2^%.2f targetSF=2^%.2f sourceSF=2^%.2f "
                                 "adjFactor=2^%.2f\n", (int)correction, std::log2(pow), std::log2(targetSF),
                         std::log2(sourceSF), std::log2(adjustmentFactor));
        if constexpr (PRINT)
            std::cout << adjustmentFactor << std::endl;
        res = adjustmentFactor;
    } else {  // THIS is only for FIXEDAUTO/FIXEDMANUAL (AdjustCiphertext)
              // Scaling down the message by a correction factor to emulate using a larger q0.
              // This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        res = std::pow((double)2.0, (double)-1.0 * (double)correction);
    }

    return res;
}

void FIDESlib::CKKS::ModRaise(Ciphertext& ctxt, const int slots, const int32_t correction, const bool prescaled,
                              const bool sparse_encaps) {
    CudaNvtxRange r(std::string{sc::current().function_name()}.substr());
    ContextData& cc = ctxt.cc;
    //------------------------------------------------------------------------------
    // RAISING THE MODULUS
    //------------------------------------------------------------------------------

    if (!prescaled) {
        assert(ctxt.getLevel() - ctxt.NoiseLevel + 1 >= 1);
    } else {
        assert(ctxt.getLevel() - ctxt.NoiseLevel + 1 == 0);
    }
    // In FLEXIBLEAUTO, raising the ciphertext to a larger number
    // of towers is a bit more complex, because we need to adjust
    // it's scaling factor to the one that corresponds to the level
    // it's being raised to.
    // Increasing the modulus
    if constexpr (PRINT) {
        cudaDeviceSynchronize();
        std::cout << "Initial ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb)
                SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
        cudaDeviceSynchronize();
        CudaCheckErrorMod;
    }
    if (ctxt.NoiseLevel == 2)
        ctxt.rescale();
    if constexpr (PRINT) {
        cudaDeviceSynchronize();
        std::cout << "Initial 2 ";
        CudaCheckErrorMod;
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb)
                SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
        std::cout << correction << std::endl;
        std::cout << std::pow((double)2.0, (double)-1.0 * (double)correction) << std::endl;
        cudaDeviceSynchronize();
        CudaCheckErrorMod;
    }

    if (cc.rescaleTechnique == CKKS::FLEXIBLEAUTO || cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT) {
        uint32_t lvl = cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT;
        // RATIONAL RESCALING: sourceSF is already dynamic (the ciphertext's own NoiseFactor);
        // only the two chain-derived quantities move to level indexing. The rescale that
        // follows is the one from the ciphertext's CURRENT level, so its factor is
        // F(getLevel()) — and the ciphertext must be at level >= 1 for that to exist, which is
        // the precondition the drop sites below enforce. The classic arms must not be
        // EVALUATED on an RR chain: sfAtLimb now throws there.
        const bool rr_     = cc.isRR();
        double targetSF    = rr_ ? cc.sfAtLevel(cc.topLevel()) : cc.sfAtLimb(cc.L - lvl * cc.compositeDegree());
        double sourceSF    = ctxt.NoiseFactor;  // ciphertext->GetScalingFactor();
        uint32_t numTowers = rr_ ? (uint32_t)cc.windowSize(ctxt.getLevel()) : (uint32_t)(ctxt.getLevel() + 1);
        // composite: the adjust's rescale drops the top d primes — divide by their product
        double modToDrop = rr_ ? (ctxt.getLevel() >= 1 ? cc.rrRescaleFactor(ctxt.getLevel()) : 1.0)
                               : cc.modReduceProduct(ctxt.getLevel());
        //cryptoParams->GetElementParams()->GetParams()[numTowers - 1]->GetModulus().ConvertToDouble();

        // in the case of FLEXIBLEAUTO, we need to bring the ciphertext to the right scale using a
        // a scaling multiplication. Note the at currently FLEXIBLEAUTO is only supported for NATIVEINT = 64.
        // So the other branch is for future purposes (in case we decide to add add the FLEXIBLEAUTO support
        // for NATIVEINT = 128.
        // Scaling down the message by a correction factor to emulate using a larger q0.
        // This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        double adjustmentFactor = (targetSF / sourceSF) * (modToDrop / sourceSF);
        double pow = std::pow((double)2.0, (double)-1.0 * (double)correction);
        adjustmentFactor *= pow;
        if (cc.isRR() && std::getenv("RR_BTS_SCALEDBG"))
            std::fprintf(stderr, "[rr_scale] adjust: correction=%d pow=2^%.2f targetSF=2^%.2f sourceSF=2^%.2f "
                                 "adjFactor=2^%.2f prescaled=%d\n", (int)correction, std::log2(pow), std::log2(targetSF),
                         std::log2(sourceSF), std::log2(adjustmentFactor), (int)prescaled);
        if constexpr (PRINT)
            std::cout << adjustmentFactor << std::endl;
        if (std::getenv("BTS_SF_DEBUG"))
            printf("[bts_sf] towers=%u log2(targetSF)=%.4f log2(sourceSF)=%.4f "
                   "log2(modToDrop)=%.4f corr=%u log2(adj)=%.4f\n",
                   numTowers, log2(targetSF), log2(sourceSF), log2(modToDrop),
                   correction, log2(adjustmentFactor));

        if (!prescaled) {
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Initial ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
            ctxt.multScalar(adjustmentFactor);
            if (cc.isRR() && std::getenv("RR_BTS_SCALEDBG"))
                std::fprintf(stderr, "[rr_scale] adjust: multScalar(adj) APPLIED\n");
            BTS_DIAG("AFTER multScalar(adj)");
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Initial ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
            //cc->EvalMultInPlace(ciphertext, adjustmentFactor);
            ctxt.rescale();
            if (cc.isRR())
                rrRequireBottom(ctxt, "adjust (non-prescaled)");
            else
                ctxt.dropToLevel(cc.compositeDegree() - 1);
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Initial ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
        } else {
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Prescale path ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
            if (ctxt.NoiseLevel == 2) {
                // deg-2 ciphertext: it owes one rescale. Classic drops to 2 levels first so
                // that the rescale lands on the bottom; under RR the ciphertext must already
                // BE at level 1, and the rescale is what takes it to 0.
                if (cc.isRR()) {
                    if (ctxt.getLevel() != 1)
                        throw std::runtime_error(
                            "RR bootstrap: prescaled deg-2 entry expects RR level 1 (one rescale above the "
                            "bottom), found level " +
                            std::to_string(ctxt.getLevel()));
                } else {
                    ctxt.dropToLevel(2 * cc.compositeDegree() - 1);
                }
                ctxt.rescale();
            } else if (cc.isRR()) {
                rrRequireBottom(ctxt, "adjust (prescaled, deg-1)");
            } else {
                ctxt.dropToLevel(cc.compositeDegree() - 1);
            }
        }
        // KNOWN GAP (next unit of work): on the PRESCALED path the correction is never applied
        // to the message -- `multScalar(adjustmentFactor)` below, which folds in
        // pow(2,-correction), sits under `if (!prescaled)` and the RR probe enters prescaled.
        // The recovery at the end still relabels the scale by 2^-correction, so the output comes
        // out exactly 2^-correction of the message (measured: 1e-5 against 0.5, correction -14).
        // Applying the multiply HERE was tried and crashes (illegal access,
        // RationalRescale.cu:203) -- the ciphertext is mid-adjustment at this point. It belongs
        // either before the level moves in this branch, or in the caller that prescales.
        ctxt.NoiseFactor = targetSF;
        btsStageProbe("post-adjust", ctxt);
    } else {  // THIS is only for FIXEDAUTO/FIXEDMANUAL (AdjustCiphertext)
              // Scaling down the message by a correction factor to emulate using a larger q0.
              // This step is needed so we could use a scaling factor of up to 2^59 with q9 ~= 2^60.
        if (!prescaled) {
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Initial ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
            ctxt.multScalar(std::pow((double)2.0, (double)-1.0 * (double)correction), false);
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Initial ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
            ctxt.rescale();
            if (cc.isRR())
                rrRequireBottom(ctxt, "adjust (non-prescaled)");
            else
                ctxt.dropToLevel(cc.compositeDegree() - 1);
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Initial ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
        } else {
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Prescale path ";
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb)
                        SWITCH(i, printThisLimb(1));
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
                CudaCheckErrorMod;
            }
            if (ctxt.NoiseLevel == 2) {
                // deg-2 ciphertext: it owes one rescale. Classic drops to 2 levels first so
                // that the rescale lands on the bottom; under RR the ciphertext must already
                // BE at level 1, and the rescale is what takes it to 0.
                if (cc.isRR()) {
                    if (ctxt.getLevel() != 1)
                        throw std::runtime_error(
                            "RR bootstrap: prescaled deg-2 entry expects RR level 1 (one rescale above the "
                            "bottom), found level " +
                            std::to_string(ctxt.getLevel()));
                } else {
                    ctxt.dropToLevel(2 * cc.compositeDegree() - 1);
                }
                ctxt.rescale();
            } else if (cc.isRR()) {
                rrRequireBottom(ctxt, "adjust (prescaled, deg-1)");
            } else {
                ctxt.dropToLevel(cc.compositeDegree() - 1);
            }
        }
    }

    if (sparse_encaps) {
        // RR joins the composite branch, and this is a COHERENCE fix, not a preference: the
        // KEYGEN side already routes RR through the MAIN context (createContextSwitchingKeys,
        // `srcCompositeDegree > 1 || srcIsRR`), while this CONSUMER still asked the helper
        // context for the key. Keys made in one context and consumed through another is the
        // illegal access that killed both sparse routes. The argument is the composite one
        // verbatim: the helper context is single-tower by construction and an RR bottom level
        // is a THREE-limb window, so it cannot host the ciphertext at all.
        if (cc.compositeDegree() > 1 || cc.isRR()) {
            // COMPOSITESCALING: M-4 as a STANDARD hybrid keyswitch in the MAIN context at the
            // composite bottom — the single-tower helper context cannot host a d-limb ct
            // (its digit tables stop at k=0; see BootstrapPrecomputation::sparse_atob).
            ctxt.keySwitch(*cc.GetBootPrecomputation(slots).sparse_atob);
        } else {
            auto& sparse_context = cc.GetBootPrecomputation(slots).sparse_context;
            auto sparse_context_use = sparse_context.lock();
            Ciphertext sparse_ctxt(sparse_context_use);
            auto& atob = CKKS::GetSecretSwitchingKey(ctxt.cc_, sparse_context_use, ctxt.keyID);

            sparse_ctxt.reinterpretContext(ctxt);
            sparse_ctxt.keySwitch(atob);
            ctxt.reinterpretContext(sparse_ctxt);
        }
    }

    //   std::cout << "Boot start " << std::endl;
    // auto ctxtDCRT = raised->GetElements();
    if constexpr (PRINT) {
        std::cout << "Adjustment 1: ";
        CudaCheckErrorMod;
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }

    ctxt.c0.INTT(cc.batch, true);

    if constexpr (PRINT) {
        CudaCheckErrorMod;
        std::cout << "Adjustment ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    //   std::cout << "Grow" << std::endl;
    // RATIONAL RESCALING (RR_PLAN Phase 2): rrModRaise IS the grow + raise. It widens the
    // level-0 window onto the TOP window (rrWidenToLevel — (c).2's no-widen assert stays
    // armed for everything else) and CRT-extends the bottom window's w residues across it.
    // The classic pair below cannot express either half: `grow` to a prefix is not an RR
    // level, and broadcastLimb0 assumes a SINGLE-limb bottom where RR has a w-limb window.
    if (cc.isRR()) {
        ctxt.c0.rrModRaise();
    } else {
    ctxt.c0.grow(cc.L - (cc.rescaleTechnique == FLEXIBLEAUTOEXT));
    //   std::cout << "Broadcast" << std::endl;
    if constexpr (PRINT) {
        CudaCheckErrorMod;
        std::cout << "Adjustment ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    if (cc.compositeDegree() > 1)
        ctxt.c0.compositeModRaise();
    else
        ctxt.c0.broadcastLimb0();
    }
    if constexpr (PRINT) {
        CudaCheckErrorMod;
        std::cout << "Adjustment ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    ctxt.c0.NTT(cc.batch, true);
    // std::cout << cc.batch << std::endl;
    if constexpr (PRINT) {
        std::cout << "ModRaise ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    ctxt.c1.INTT(cc.batch, true);
    if constexpr (PRINT) {
        std::cout << "Adjustment c1 ";
        for (auto& j : ctxt.c1.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    //  std::cout << "Grow" << std::endl;
    // RATIONAL RESCALING (RR_PLAN Phase 2): rrModRaise IS the grow + raise. It widens the
    // level-0 window onto the TOP window (rrWidenToLevel — (c).2's no-widen assert stays
    // armed for everything else) and CRT-extends the bottom window's w residues across it.
    // The classic pair below cannot express either half: `grow` to a prefix is not an RR
    // level, and broadcastLimb0 assumes a SINGLE-limb bottom where RR has a w-limb window.
    if (cc.isRR()) {
        ctxt.c1.rrModRaise();
    } else {
    ctxt.c1.grow(cc.L - (cc.rescaleTechnique == FLEXIBLEAUTOEXT));
    //  std::cout << "Broadcast" << std::endl;
    if constexpr (PRINT) {
        std::cout << "Adjustment c1  ";
        for (auto& j : ctxt.c1.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    if (cc.compositeDegree() > 1)
        ctxt.c1.compositeModRaise();
    else
        ctxt.c1.broadcastLimb0();
    }
    if constexpr (PRINT) {
        std::cout << "Adjustment c1";
        for (auto& j : ctxt.c1.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
    ctxt.c1.NTT(cc.batch, true);
    if constexpr (PRINT) {
        std::cout << "Adjustment c1";
        for (auto& j : ctxt.c1.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }

    if (sparse_encaps) {
        // RR joins the composite branch here too — the return leg of the same key pair, and it
        // has to match the outbound one above or the ciphertext comes back under the wrong key.
        if (cc.compositeDegree() > 1 || cc.isRR()) {
            // COMPOSITESCALING: M-2 back to the dense key, MAIN-context standard hybrid key.
            ctxt.keySwitch(*cc.GetBootPrecomputation(slots).sparse_btoa);
        } else {
            auto& sparse_context = cc.GetBootPrecomputation(slots).sparse_context;

            auto sparse_context_use = sparse_context.lock();

            auto& btoa = CKKS::GetSecretSwitchingKey(sparse_context_use, ctxt.cc_, ctxt.keyID);

            ctxt.keySwitch(btoa);
        }
    }

    ctxt.slots = cc.N / 2;
}
