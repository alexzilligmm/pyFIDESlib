//
// Created by carlosad on 4/12/24.
//

#include <atomic>
#include <cstdlib>
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
// btsStageProbe checkpoint (pre-CtS / post-CtS / pre-StC / post-StC / end) also deposits a
// full ciphertext clone the caller can download+decrypt offline. Zero cost when null.
std::vector<std::pair<std::string, std::shared_ptr<FIDESlib::CKKS::Ciphertext>>>*
    FIDESlib::CKKS::g_btsStageStash = nullptr;


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
        auto c = std::make_shared<FIDESlib::CKKS::Ciphertext>(ctxt.cc_);
        c->copy(ctxt);
        FIDESlib::CKKS::g_btsStageStash->emplace_back(stage, std::move(c));
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
    uint64_t q = cc.prime[0].p;
    double qDouble = (double)q;  //q.ConvertToDouble();
    // COMPOSITESCALING: level 0 spans compositeDegree primes — the bootstrap's q0 is their
    // PRODUCT (~2^54 on a 2x27-bit chain). Everything downstream (deg, correction, pre/post,
    // the ModRaise CRT lift) is derived from it.
    for (int j_ = 1; j_ < cc.compositeDegree(); ++j_)
        qDouble *= (double)cc.prime[j_].p;

    if constexpr (PRINT) {
        std::cout << "q: " << q << " ";
        std::cout << qDouble << std::endl;
    }
    const auto p = cc.param.raw->p;  //cryptoParams->GetPlaintextModulus();
    double powP = pow(2, p);

    if constexpr (PRINT) {
        std::cout << "p: " << p << std::endl;
    }
    int32_t deg = std::round(std::log2(qDouble / powP));
    // Guard restored (was commented out upstream): deg = q0_bits - scale_bits
    // must not exceed the correction factor (OpenFHE auto = 9), or the uint32
    // subtraction below underflows and corFactor = 1 << garbage poisons every
    // bootstrap SILENTLY (cost us a 6-config param sweep of tok0 garbage).
    if (deg > static_cast<int32_t>(effCorrectionFactor(cc, slots))) {
        throw std::runtime_error(
            "Bootstrap: deg=log2(q0/2^p)=" + std::to_string(deg) +
            " exceeds correctionFactor=" +
            std::to_string(effCorrectionFactor(cc, slots)) +
            " (uint32 underflow); pick q0_bits - scale_bits <= correctionFactor.");
    }
    uint32_t correction = effCorrectionFactor(cc, slots) - deg;
    if constexpr (PRINT)
        std::cout << effCorrectionFactor(cc, slots) << " " << deg << std::endl;
    double post = std::pow(2, static_cast<double>(deg));

    double pre = 1. / post;
    uint64_t scalar = std::llround(post);

    // Mixed-size chain (see OpenFHE ckksrns-fhe.cpp, same gate): the uniform
    // identity sf[0] ~ 2^p * 2^deg does not hold; follow the COMPOSITESCALING
    // constants: pre = sf[0]/q0 input normalization, no integer 2^deg recovery
    // (the CPU-precomputed StC matrices carry scaleDec = q0/sf[0]).
    bool mixedChain = std::fabs(std::log2(cc.sfAtLimb(cc.L) * post / qDouble)) > 0.5;
    // COMPOSITESCALING always uses the sf/q0 normalization (OpenFHE: pre = sf[0]/qDouble,
    // no integer 2^deg recovery) — the same constants the mixed-chain arm implements.
    if (mixedChain || cc.compositeDegree() > 1) {
        pre    = cc.sfAtLimb(cc.L) / qDouble;
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
        // Free per-call input pre-scale (Context.cuh btsPreScale): rides the arbitrary
        // double the input is multiplied by anyway. Any restore is the caller's business.
        constantEvalMult *= cc.getBtsPreScale();

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
    uint64_t corFactor = (uint64_t)1 << std::llround(correction);
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
    // Mixed-size chain: realize the pending StC rescale so the output lands
    // deg-1 exactly on the per-level table at the data scale (the lazy deg-2
    // state does not match ScalingFactorRealBig there). Uniform unchanged.
    if (mixedChain && ctxt.NoiseLevel == 2)
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

// ── CUDA GRAPH BRING-UP (add.166 add.69, FIDESLIB_BTS_GRAPH) ────────────────────────────────
// Why this is being reopened after add.60 closed it: the close was correct on its own evidence
// (bootstrap interiors are 91.4% GPU-busy, so the ceiling is small) but three things changed.
//   * add.62 measured the REAL intra-bootstrap idle unprofiled at ~1.0-1.3 ms/bts = ~0.6-0.75
//     s/token, which after add.61 landed is the LARGEST remaining addressable item;
//   * add.62 also REFUTED the allocator/event-overhead explanation, leaving the arithmetic
//     2826 launches x ~0.35 us ~= 1.0 ms — residual per-launch dispatch gap, which is precisely
//     and only what a graph removes;
//   * FIDESLIB_SCALAR_DEV_MEMO (add.62) already removes multScalar's per-call PAGEABLE H2D, one
//     of the capture blockers the add.61 design review identified.
//
// mode 0 = off, byte-identical eager (default).
// mode 2 = DRY CAPTURE: capture on a CLONE, report the outcome, discard the graph, then run the
//          real bootstrap eagerly. A clone is required because work does NOT execute during
//          capture — probing the live ciphertext would leave it garbage AND double-apply the
//          host-side metadata updates (NoiseFactor/level/slots), which replay never re-runs.
//          This mode answers "is capture even possible, and with what error" for one build.
static int btsGraphMode() {
    static const int v = [] {
        const char* e = std::getenv("FIDESLIB_BTS_GRAPH");
        return (e && *e) ? std::atoi(e) : 0;
    }();
    return v;
}

namespace FIDESlib::CKKS { static void BootstrapEager(Ciphertext& ctxt, int slots, bool prescaled); }
using FIDESlib::CKKS::BootstrapEager;

void FIDESlib::CKKS::Bootstrap(Ciphertext& ctxt, const int slots, const bool prescaled) {
    if (btsGraphMode() == 2) {   // == not >=: mode 3 must reach its own branch below
        // WARM UP FIRST. Several allocations in this library are lazily initialised on first use
        // and are capture-hostile: `modup_ksk_moddown_mgpu`'s static `signals` does
        // cudaFreeHost + cudaHostAlloc + cudaDeviceSynchronize (RNSPoly.cpp:1474-1510), all of
        // which are "operation not permitted when stream is capturing". They run ONCE, so the fix
        // is to let a few bootstraps run eagerly before probing rather than to touch that code.
        static std::atomic<int> calls{0};
        if (calls.fetch_add(1) < 3) { BootstrapEager(ctxt, slots, prescaled); return; }
        static std::atomic<int> reported{0};
        static std::atomic<int> ok{0}, fail{0};
        Ciphertext probe(ctxt.cc_);
        probe.copy(ctxt);
        // Capture the PROBE's stream, not ctxt's. The clone owns its own partition streams, so
        // capturing ctxt's produced a valid but EMPTY graph (nodes=0) — the tell that the work
        // went somewhere the capture was not watching.
        cudaStream_t s = probe.c0.GPU[0].getS().ptr();
        cudaGraph_t g = nullptr;
        static cudaEvent_t fork_ev = [] {
            cudaEvent_t e = nullptr;
            cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
            return e;
        }();
        std::vector<cudaStream_t> forked;
        cudaError_t eb = cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed);
        if (eb == cudaSuccess) {
            // Pull every live FIDESlib stream INTO the capture, else the first cross-stream
            // Stream::wait is cudaErrorStreamCaptureIsolation ("dependency created on uncaptured
            // work in another stream", CudaUtils.cu:235) — which is how the previous attempt died.
            FIDESlib::captureForkAll(s, forked, fork_ev);
            std::fprintf(stderr, "[bts_graph] forked %zu stream(s) into the capture\n",
                         forked.size());
            BootstrapEager(probe, slots, prescaled);
            FIDESlib::captureJoinAll(s, forked, fork_ev);   // un-joined fork => CaptureUnjoined
            cudaError_t ee = cudaStreamEndCapture(s, &g);
            size_t nodes = 0;
            if (ee == cudaSuccess && g) cudaGraphGetNodes(g, nullptr, &nodes);
            if (ee == cudaSuccess) ++ok; else ++fail;
            if (reported.fetch_add(1) < 12)
                std::fprintf(stderr,
                             "[bts_graph] dry begin=%s end=%s nodes=%zu forked=%zu slots=%d lvl=%d\n",
                             cudaGetErrorName(eb), cudaGetErrorName(ee), nodes, forked.size(),
                             slots, (int)ctxt.c0.getLevel());
            if (g) cudaGraphDestroy(g);
        } else {
            ++fail;
            if (reported.fetch_add(1) < 12)
                std::fprintf(stderr, "[bts_graph] dry begin=%s (capture refused)\n",
                             cudaGetErrorName(eb));
        }
        cudaGetLastError();   // the probe's failures are DIAGNOSTIC; never leak into the real run
        BootstrapEager(ctxt, slots, prescaled);
        return;
    }
    if (btsGraphMode() == 3) {
        // ── REPLAY, minimal proof (add.166 add.70) ──────────────────────────────────────────
        // Capture on a clone, instantiate, launch ONCE on that same clone, copy the result back.
        // Why capture-then-replay on the SAME object is self-consistent, and why it sidesteps
        // both of the hard problems for this first step:
        //   * BAKED POINTERS: the graph is replayed on exactly the buffers it captured, so the
        //     addresses are correct by construction. (A cache across DIFFERENT ciphertexts is
        //     the next problem, not this one.)
        //   * HOST METADATA: during capture the host code RUNS — only GPU work is deferred — so
        //     the clone already carries the post-bootstrap level/NoiseFactor/slots, and its
        //     buffers still hold the INPUT. Launching then fills them with the real result.
        // Cost is deliberately terrible (capture + instantiate + destroy every call): this step
        // answers "does a replayed bootstrap produce the right ANSWER", nothing else. The
        // harness's own [bts_prof] err_max is the gate — a broken replay blows it up.
        static std::atomic<int> calls3{0};
        if (calls3.fetch_add(1) < 3) { BootstrapEager(ctxt, slots, prescaled); return; }
        static std::atomic<int> shown{0};
        static cudaEvent_t fev = [] {
            cudaEvent_t e = nullptr;
            cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
            return e;
        }();
        Ciphertext probe(ctxt.cc_);
        probe.copy(ctxt);
        cudaStream_t s = probe.c0.GPU[0].getS().ptr();
        std::vector<cudaStream_t> forked;
        cudaGraph_t g = nullptr;
        cudaGraphExec_t exec = nullptr;
        bool ok = false;
        const bool trace = shown.load() < 2;
        auto st = [&](const char* w) {
            if (trace) { std::fprintf(stderr, "[bts_graph] stage=%s\n", w); std::fflush(stderr); }
        };
        cudaGetLastError();   // clear anything stale so what we see below is OURS
        st("begin");
        if (cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed) == cudaSuccess) {
            FIDESlib::captureForkAll(s, forked, fev);
            st("body");
            BootstrapEager(probe, slots, prescaled);
            st("join");
            FIDESlib::captureJoinAll(s, forked, fev);
            cudaError_t ee = cudaStreamEndCapture(s, &g);
            if (trace) std::fprintf(stderr, "[bts_graph] end=%s\n", cudaGetErrorName(ee));
            if (ee == cudaSuccess && g) {
                if (trace) {   // node-type census: names the population that must be fixed
                    size_t n = 0;
                    cudaGraphGetNodes(g, nullptr, &n);
                    std::vector<cudaGraphNode_t> nodes(n);
                    cudaGraphGetNodes(g, nodes.data(), &n);
                    size_t k = 0, mc = 0, ml = 0, ev = 0, oth = 0;
                    for (size_t i = 0; i < n; ++i) {
                        cudaGraphNodeType t{};
                        cudaGraphNodeGetType(nodes[i], &t);
                        if (t == cudaGraphNodeTypeKernel) ++k;
                        else if (t == cudaGraphNodeTypeMemcpy) ++mc;
                        else if (t == cudaGraphNodeTypeMemAlloc || t == cudaGraphNodeTypeMemFree) ++ml;
                        else if (t == cudaGraphNodeTypeEventRecord || t == cudaGraphNodeTypeWaitEvent) ++ev;
                        else ++oth;
                    }
                    std::fprintf(stderr,
                                 "[bts_graph] nodes=%zu kernel=%zu MEMCPY=%zu memalloc/free=%zu "
                                 "event=%zu other=%zu\n", n, k, mc, ml, ev, oth);
                }
                cudaError_t ei = cudaGraphInstantiateWithFlags(&exec, g, 0);
                if (trace) std::fprintf(stderr, "[bts_graph] inst=%s\n", cudaGetErrorName(ei));
                if (ei == cudaSuccess && exec) {
                    cudaError_t el = cudaGraphLaunch(exec, s);
                    cudaError_t es = cudaStreamSynchronize(s);
                    if (trace)
                        std::fprintf(stderr, "[bts_graph] launch=%s sync=%s\n",
                                     cudaGetErrorName(el), cudaGetErrorName(es));
                    if (el == cudaSuccess && es == cudaSuccess) ok = true;
                }
            }
        }
        if (g) cudaGraphDestroy(g);
        if (exec) cudaGraphExecDestroy(exec);
        if (shown.fetch_add(1) < 6)
            std::fprintf(stderr, "[bts_graph] replay %s slots=%d\n", ok ? "OK" : "FAILED", slots);
        if (ok) {
            ctxt.copy(probe);       // the replayed result becomes the answer -> err_max judges it
            cudaGetLastError();
            return;
        }
        cudaGetLastError();
        BootstrapEager(ctxt, slots, prescaled);   // replay refused: fall back, never fail the run
        return;
    }
    BootstrapEager(ctxt, slots, prescaled);
}

namespace FIDESlib::CKKS {
static void BootstrapEager(Ciphertext& ctxt, const int slots, const bool prescaled) {
    CudaNvtxRange r(std::string{sc::current().function_name()});

    assert(slots >= ctxt.slots);
    int old_slots = ctxt.slots;

    FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
    ContextData& cc = ctxt.cc;
    Ciphertext aux(cc_);
    bool isLT = cc.GetBootPrecomputation(slots).LT.slots == slots;

    /////////////////////////////////////////////////////////////////////
    //NativeInteger q = elementParamsRaisedPtr->GetParams()[0]->GetModulus().ConvertToInt();
    uint64_t q = cc.prime[0].p;
    double qDouble = (double)q;  //q.ConvertToDouble();
    // COMPOSITESCALING: level 0 spans compositeDegree primes — the bootstrap's q0 is their
    // PRODUCT (~2^54 on a 2x27-bit chain). Everything downstream (deg, correction, pre/post,
    // the ModRaise CRT lift) is derived from it.
    for (int j_ = 1; j_ < cc.compositeDegree(); ++j_)
        qDouble *= (double)cc.prime[j_].p;

    if constexpr (PRINT) {
        std::cout << "q: " << q << " ";
        std::cout << qDouble << std::endl;
    }
    const auto p = cc.param.raw->p;  //cryptoParams->GetPlaintextModulus();
    double powP = pow(2, p);

    if constexpr (PRINT) {
        std::cout << "p: " << p << std::endl;
    }
    int32_t deg = std::round(std::log2(qDouble / powP));
    // Guard restored (was commented out upstream): deg = q0_bits - scale_bits
    // must not exceed the correction factor (OpenFHE auto = 9), or the uint32
    // subtraction below underflows and corFactor = 1 << garbage poisons every
    // bootstrap SILENTLY (cost us a 6-config param sweep of tok0 garbage).
    if (deg > static_cast<int32_t>(effCorrectionFactor(cc, slots))) {
        throw std::runtime_error(
            "Bootstrap: deg=log2(q0/2^p)=" + std::to_string(deg) +
            " exceeds correctionFactor=" +
            std::to_string(effCorrectionFactor(cc, slots)) +
            " (uint32 underflow); pick q0_bits - scale_bits <= correctionFactor.");
    }
    uint32_t correction = effCorrectionFactor(cc, slots) - deg;
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
    bool mixedChain = std::fabs(std::log2(cc.sfAtLimb(cc.L) * post / qDouble)) > 0.5;
    // COMPOSITESCALING always uses the sf/q0 normalization (OpenFHE: pre = sf[0]/qDouble,
    // no integer 2^deg recovery) — the same constants the mixed-chain arm implements.
    if (mixedChain || cc.compositeDegree() > 1) {
        pre    = cc.sfAtLimb(cc.L) / qDouble;
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

        // Free per-call input pre-scale (Context.cuh btsPreScale): rides the arbitrary
        // double the input is multiplied by anyway. Any restore is the caller's business.
        constantEvalMult *= cc.getBtsPreScale();

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
    uint64_t corFactor = (uint64_t)1 << std::llround(correction);
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
    // Mixed-size chain: realize the pending StC rescale so the output lands
    // deg-1 exactly on the per-level table at the data scale (the lazy deg-2
    // state does not match ScalingFactorRealBig there). Uniform unchanged.
    if (mixedChain && ctxt.NoiseLevel == 2)
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
}  // namespace FIDESlib::CKKS

double FIDESlib::CKKS::GetPreScaleFactor(Context& cc_, int slots) {
    ContextData& cc = *cc_;
    SetCurrentContext(cc_);
    /////////////////////////////////////////////////////////////////////
    //NativeInteger q = elementParamsRaisedPtr->GetParams()[0]->GetModulus().ConvertToInt();
    uint64_t q = cc.prime[0].p;
    double qDouble = (double)q;  //q.ConvertToDouble();
    // COMPOSITESCALING: level 0 spans compositeDegree primes — the bootstrap's q0 is their
    // PRODUCT (~2^54 on a 2x27-bit chain). Everything downstream (deg, correction, pre/post,
    // the ModRaise CRT lift) is derived from it.
    for (int j_ = 1; j_ < cc.compositeDegree(); ++j_)
        qDouble *= (double)cc.prime[j_].p;

    if constexpr (PRINT) {
        std::cout << "q: " << q << " ";
        std::cout << qDouble << std::endl;
    }
    const auto p = cc.param.raw->p;  //cryptoParams->GetPlaintextModulus();
    double powP = pow(2, p);

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
    uint32_t correction = effCorrectionFactor(cc, slots) - deg;

    double res = 0.0;
    if (cc.rescaleTechnique == CKKS::FLEXIBLEAUTO || cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT) {
        const int d_ = cc.compositeDegree();
        uint32_t lvl = cc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT;
        double targetSF = cc.sfAtLimb(cc.L - lvl * d_);
        // composite: the pre-raise ciphertext sits at 2 LEVELS = 2d limbs; its scale lives at
        // limb 2d-1 and the adjust's rescale drops the top d primes (their product).
        double sourceSF = cc.sfAtLimb(2 * d_ - 1);  // ciphertext->GetScalingFactor();
        uint32_t numTowers = 2 * d_;                // ciphertext->GetElements()[0].GetNumOfElements();
        double modToDrop = cc.modReduceProduct(2 * d_ - 1);
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

void FIDESlib::CKKS::ModRaise(Ciphertext& ctxt, const int slots, const uint32_t correction, const bool prescaled,
                              const bool sparse_encaps) {
    CudaNvtxRange r(std::string{sc::current().function_name()}.substr());
    ContextData& cc = ctxt.cc;
    btsStageProbe("MR-entry", ctxt);
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
    // Entry state, captured BEFORE the pending-rescale drain below: the [bts_sf] line needs
    // to say what came IN, not what the drain left. deg=2 inputs are the n32 norm (1648 of
    // 1687 probed values), so "which deg was this bootstrap handed" is the first question.
    const int entry_noise_level = ctxt.NoiseLevel;
    const int entry_level = ctxt.getLevel();
    // FLEXIBLEAUTO's standing invariant is NoiseFactor == sfAtLimb(level). `drift` is how many
    // BITS it is violated by — and (see the [bts_sf] note below) exactly the power of two the
    // adjust then scales the message by. Read ScalingFactorReal directly, NOT sfAtLimb(): the
    // latter aborts on an off-grid level, and off-grid is one of the states we want to SEE.
    if (std::getenv("BTS_SF_DEBUG")) {
        const bool grid = ((cc.L - entry_level) % cc.compositeDegree()) == 0;
        const double sfl = cc.param.ScalingFactorReal[entry_level];
        fprintf(stderr, "[bts_entry] lvl=%d deg=%d on_grid=%d prescaled=%d log2NF=%.4f "
                        "log2sf=%.4f drift=%.4f\n",
                entry_level, entry_noise_level, (int)grid, (int)prescaled,
                log2(ctxt.NoiseFactor), grid ? log2(sfl) : NAN,
                grid ? log2(ctxt.NoiseFactor / sfl) : NAN);
        fflush(stderr);
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
        double targetSF = cc.sfAtLimb(cc.L - lvl * cc.compositeDegree());
        double sourceSF = ctxt.NoiseFactor;        // ciphertext->GetScalingFactor();
        uint32_t numTowers = ctxt.getLevel() + 1;  // ciphertext->GetElements()[0].GetNumOfElements();
        // composite: the adjust's rescale drops the top d primes — divide by their product
        double modToDrop = cc.modReduceProduct(ctxt.getLevel());
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
        if constexpr (PRINT)
            std::cout << adjustmentFactor << std::endl;
        if (std::getenv("BTS_SF_DEBUG")) {
            // encSF is the scale the multScalar below actually encodes `adjustmentFactor` at
            // (Ciphertext::multScalarNoPrecheck -> cc.ElemForEvalMult(c0.getLevel(), c) and
            // NoiseFactor *= cc.sfAtLimb(getLevel())). The adjustmentFactor formula divides by
            // sourceSF TWICE, which is only equivalent to dividing by (sourceSF * encSF) when
            // the FLEXIBLEAUTO invariant NoiseFactor == sfAtLimb(level) holds at this point.
            // `skew` is exactly the factor a violated invariant multiplies the message by.
            const bool on_grid = ((cc.L - ctxt.getLevel()) % cc.compositeDegree()) == 0;
            const double encSF = on_grid ? cc.sfAtLimb(ctxt.getLevel()) : 0.0;
            fprintf(stderr,
                    "[bts_sf] entry_lvl=%d entry_deg=%d lvl=%d towers=%u "
                    "log2(targetSF)=%.4f log2(sourceSF)=%.4f log2(encSF)=%.4f "
                    "log2(skew=sourceSF/encSF)=%.4f log2(modToDrop)=%.4f corr=%u log2(adj)=%.4f\n",
                    entry_level, entry_noise_level, ctxt.getLevel(), numTowers,
                    log2(targetSF), log2(sourceSF), on_grid ? log2(encSF) : NAN,
                    on_grid ? log2(sourceSF / encSF) : NAN, log2(modToDrop),
                    correction, log2(adjustmentFactor));
            fflush(stderr);
        }

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
                ctxt.dropToLevel(2 * cc.compositeDegree() - 1);
                ctxt.rescale();
            } else {
                ctxt.dropToLevel(cc.compositeDegree() - 1);
            }
        }
        ctxt.NoiseFactor = targetSF;
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
                ctxt.dropToLevel(2 * cc.compositeDegree() - 1);
                ctxt.rescale();
            } else {
                ctxt.dropToLevel(cc.compositeDegree() - 1);
            }
        }
    }

    btsStageProbe("MR-bottom", ctxt);
    if (sparse_encaps) {
        if (cc.compositeDegree() > 1) {
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
    btsStageProbe("MR-atob", ctxt);

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

    btsStageProbe("MR-raised", ctxt);
    if (sparse_encaps) {
        if (cc.compositeDegree() > 1) {
            // COMPOSITESCALING: M-2 back to the dense key, MAIN-context standard hybrid key.
            ctxt.keySwitch(*cc.GetBootPrecomputation(slots).sparse_btoa);
        } else {
            auto& sparse_context = cc.GetBootPrecomputation(slots).sparse_context;

            auto sparse_context_use = sparse_context.lock();

            auto& btoa = CKKS::GetSecretSwitchingKey(sparse_context_use, ctxt.cc_, ctxt.keyID);

            ctxt.keySwitch(btoa);
        }
    }

    btsStageProbe("MR-btoa", ctxt);
    ctxt.slots = cc.N / 2;
}
