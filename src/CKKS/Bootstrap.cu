//
// Created by carlosad on 4/12/24.
//

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>
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

// Graph-parameter tags (add.166 add.73): the two scalars the correction factor enters through.
constexpr int kBtsParamRaise = 0;    // ModRaise's multScalar(adjustmentFactor), adj *= 2^-c
constexpr int kBtsParamRestore = 1;  // multIntScalar(ctxt, 1 << c)
// What the RAISE stage used at capture, so a replay can rescale it for a different factor:
// adj_new = adj_captured * 2^(corr_captured - corr_new), at the same level (fixed by the shape).
struct BtsRaiseParam {
    int level = -1;
    double adjustment = 0;
    uint32_t correction = 0;
    // The two SHAPE-FIXED terms of adjustmentFactor (ModRaise):
    //   adj = (targetSF / NoiseFactor) * (modToDrop / NoiseFactor) * 2^-correction
    // targetSF and modToDrop depend only on the level, which the key pins, so recording them lets
    // a replay rebuild the scalar exactly for ANY NoiseFactor and ANY correction factor. That is
    // what takes both of them out of the cache key instead of multiplying the shape count.
    double targetSF = 0;
    double modToDrop = 0;
    double sourceSF = 0;   // ctxt.NoiseFactor AS USED by ModRaise (post pending-rescale drain)
    bool valid = false;
};
BtsRaiseParam g_bts_raise_param;
// The RESTORE site sits after EvalMod, at a DIFFERENT (lower) level than the raise. multIntScalar
// sizes its residue vector with ctxt.getLevel() THERE, so rebuilding it with the raise's level
// produced the wrong length and the wrong primes — measured as full detonation (KL 62-64) the
// moment the tag started firing, versus KL 0.78 while it was inert.
int g_bts_restore_level = -1;
inline void btsRecordRestoreLevel(int level) { g_bts_restore_level = level; }
inline void btsRecordRaiseParam(int level, double adj, uint32_t corr, double targetSF, double modToDrop,
                                double sourceSF) {
    g_bts_raise_param = BtsRaiseParam{level, adj, corr, targetSF, modToDrop, sourceSF, true};
}


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

namespace FIDESlib::CKKS {


static void BootstrapEager(Ciphertext& ctxt, int slots, bool prescaled);
static bool BootstrapCached(Ciphertext& ctxt, int slots, bool prescaled);
}  // namespace FIDESlib::CKKS
using FIDESlib::CKKS::BootstrapCached;
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
    if (btsGraphMode() == 4) {
        if (BootstrapCached(ctxt, slots, prescaled))
            return;
        BootstrapEager(ctxt, slots, prescaled);   // any doubt at all -> eager; never fail the run
        return;
    }
    BootstrapEager(ctxt, slots, prescaled);
}

namespace FIDESlib::CKKS {

// ── CACHED REPLAY (add.166 add.73, FIDESLIB_BTS_GRAPH=4) ────────────────────────────────────
// Mode 3 proved a captured bootstrap replays correctly, but it captures and instantiates on EVERY
// call, so it can never be a speed lever. This is the cached form: capture ONCE per shape, replay
// the exec for every later bootstrap of that shape.
//
// ⭐ WHY A DEDICATED SLOT, and not "replay on the caller's ciphertext". A graph bakes the pointer
// arguments of all 2675 kernel nodes at capture. Replaying on a different ciphertext would have
// every one of them addressing the ORIGINAL ciphertext's limbs. The two ways out are patching the
// nodes (cudaGraphExecKernelNodeSetParams x2675 per replay — far more work than the ~1 ms of
// dispatch gap it would save) or making the buffers CONSTANT. This takes the second: the cache
// owns one ciphertext per shape, and every bootstrap of that shape is copy-in -> replay -> copy-out.
// Two limb copies against a 29 ms bootstrap is noise, and the pointers are correct by construction
// — the same reason mode 3 works.
//
// ⚠️ copy-in MUST NOT RESIZE. RNSPoly::copy grows/drops, which reallocates and moves the very
// addresses the graph baked. copyShallow is the documented "copy contents without resizing"
// primitive and is what keeps the slot's buffers fixed.
//
// ⚠️ REPLAY RUNS NO HOST CODE, so the bootstrap's host-side metadata transitions (level,
// NoiseLevel, NoiseFactor, slots) never happen. They are a pure function of the input shape, so
// they are recorded once at capture and re-applied on every hit. Mode 3 sidestepped this only
// because host code DID run during its capture.
namespace {

struct BtsShapeKey {
    int slots_arg;
    int prescaled;
    int level;
    int noise_level;
    int ct_slots;
    // ⚠️ NoiseFactor and the correction factor were BOTH tried as key fields and both are wrong.
    // They change only the VALUE of two scalars whose kernels are identical, so they belong in the
    // graph as PARAMETERS, not in its identity — rebuilt into the pinned arena before each replay
    // (see the re-point block on the hit path). Keying on them is what made this unusable:
    // NoiseFactor is float noise that never repeats (~0% hit rate), and cf takes 11 values on this
    // plan, together exhausting 80 GB at 41 shapes. Omitting them WITHOUT the rebuild is equally
    // wrong and much quieter — that ran to completion at ~94% hit and returned KL 62.11,
    // top5_overlap 0/5, z_mass 3.1e51.
    bool operator<(const BtsShapeKey& o) const {
        return std::tie(slots_arg, prescaled, level, noise_level, ct_slots) <
               std::tie(o.slots_arg, o.prescaled, o.level, o.noise_level, o.ct_slots);
    }
};

struct BtsCacheEntry {
    std::unique_ptr<Ciphertext> slot;
    cudaGraphExec_t exec = nullptr;
    bool poisoned = false;              // capture failed once => never retry this shape
    int in_level = 0, in_noise = 0, in_slots = 0;
    int out_level = 0, out_noise = 0, out_slots = 0;
    double out_nf = 0;
    // ⭐ The buffers the graph's ENTRY kernels read, captured BEFORE BeginCapture. The bootstrap
    // CHANGES the limb set (drop/grow), so the slot's structure after capture is not the structure
    // it had going in — fingerprinting the post-capture state and copying into that was this
    // design's first bug (err_max 1126: the cache mechanically worked and the answer was garbage).
    // Copy-in writes straight into these addresses and the slot is never restructured again.
    std::vector<std::pair<void*, size_t>> in_bufs;
    std::vector<const void*> fingerprint;   // post-capture structure; must not drift under us
    // ⭐ EXCLUSIVE ownership of the aux polys this graph baked addresses for (add.166 add.73).
    // The bootstrap draws its temporaries from ContextData's shared LIFO and returns them at the
    // end. A cached exec still writes to those exact buffers on every replay, so if the pool hands
    // them to the next ciphertext the replay stomps live data — which is what decode reported as
    // 'an illegal memory access was encountered'. Holding them here retires them from the pool for
    // the exec's lifetime. Costs one aux working set per cached shape.
    // Tagged graph PARAMETERS: the arena slots the correction-factor scalars were staged into,
    // plus what the capture used, so a replay can rewrite them for a different factor.
    std::vector<std::pair<void*, size_t> > raise_slots, restore_slots;
    BtsRaiseParam raise_param;
    uint32_t cap_correction = 0;   // `correction` the capture used
    uint32_t cap_eff_cf = 0;       // effCorrectionFactor(cc, slots) at capture
    double cap_prescale = 1.0;     // cc.getBtsPreScale() at capture — see the guard on the hit path
    double cap_entry_nf = 0;       // ctxt.NoiseFactor at BootstrapCached entry (PRE-drain)
    int cap_restore_level = -1;    // ctxt.getLevel() at the RESTORE site (post-EvalMod)
    // ⭐ PER-ENTRY working set: the aux polys and keyswitch workspaces whose addresses THIS graph
    // baked, owned for the exec's lifetime so nothing else is ever handed them (add.166 add.73).
    // Sharing ONE set across all entries was tried to save memory and segfaults: each capture
    // re-partitions the reserved pool (the slot ciphertexts draw from it too), so a later capture
    // hands a buffer to one graph that an earlier graph still writes.
    // Per-entry was only unaffordable while NoiseFactor and the correction factor were in the key
    // (41 shapes x ~2 GB exhausted 80.8 GB). With both moved to graph PARAMETERS the key is just
    // the shape, so the entry count is small enough for private sets to fit.
    std::vector<RNSPoly> owned_aux;
    std::unique_ptr<ContextData::KsWorkspaceSet> ks_ws;
    uint64_t hits = 0;
};

std::atomic<uint64_t> g_cache_hits{0};
std::mutex bts_cache_mtx;
std::map<BtsShapeKey, BtsCacheEntry> bts_cache;

BtsShapeKey shapeOf(const Ciphertext& ct, int slots_arg, bool prescaled) {
    // NEITHER NoiseFactor NOR the correction factor is part of the identity (add.166 add.73).
    // Both only change the VALUE of two scalars whose kernels are identical, and both are rebuilt
    // into the pinned arena before each replay. Keying on them was what exploded the shape count
    // (NoiseFactor is float noise that never repeats; cf takes 11 values) and exhausted VRAM.
    return BtsShapeKey{slots_arg, prescaled ? 1 : 0, ct.getLevel(), ct.NoiseLevel, ct.slots};
}

std::vector<const void*> fingerprintOf(const Ciphertext& ct) {
    std::vector<const void*> v;
    ct.c0.appendLimbPointers(v);
    ct.c1.appendLimbPointers(v);
    return v;
}

}  // namespace

static bool BootstrapCached(Ciphertext& ctxt, const int slots, const bool prescaled) {
    // Single-threaded by construction for now: capture is a property of the STREAM, not the thread,
    // so a concurrent worker touching a forked stream would have its work swept into the graph.
    // The lock serialises cache users; excluding the residency worker is still owed.
    std::lock_guard<std::mutex> guard(bts_cache_mtx);

    Context& cc_ = ctxt.cc_;
    ContextData& cc = ctxt.cc;

    const BtsShapeKey key = shapeOf(ctxt, slots, prescaled);
    BtsCacheEntry& e = bts_cache[key];
    if (e.poisoned)
        return false;

    static std::atomic<int> warm{0};
    static std::atomic<int> reported{0};

    if (e.exec == nullptr) {
        // Lazy statics in this library are capture-hostile and fire once (RNSPoly.cpp's `signals`
        // does cudaHostAlloc + cudaDeviceSynchronize). Let a few bootstraps run eagerly first.
        if (warm.fetch_add(1) < 3)
            return false;

        // Swap the general aux pool out for the RESERVED one, and install the shared bootstrap
        // keyswitch workspaces. Both are restored on every exit path by the guard.
        // Drain the pool so the warm+capture below build FRESH aux polys, install this entry's
        // private keyswitch workspaces, and on the way out keep both and put the general pool back.
        std::vector<RNSPoly> saved_pool = cc.takeAuxilarPool();
        e.ks_ws = std::make_unique<ContextData::KsWorkspaceSet>();
        cc.setKsWorkspaceOverride(e.ks_ws.get());
        struct WsGuard {
            ContextData& c;
            BtsCacheEntry& ent;
            std::vector<RNSPoly>& saved;
            ~WsGuard() {
                c.setKsWorkspaceOverride(nullptr);
                ent.owned_aux = c.takeAuxilarPool();   // exactly what this capture used -> ours
                c.restoreAuxilarPool(std::move(saved));
            }
        } _ws_guard{cc, e, saved_pool};

        // ⭐ WARM THE PRIVATE WORKING SET WITH ONE EAGER BOOTSTRAP, BEFORE BeginCapture.
        // The workspaces above are created lazily on first touch (make_unique +
        // generateDecompAndDigit + generateSpecialLimbs), and the aux polys are built on demand
        // too. Letting that happen INSIDE the capture records buffer materialisation rather than
        // work, and it failed with 'misaligned address' before a single capture completed.
        // A throwaway eager bootstrap materialises every workspace, populates the drained aux pool
        // with exactly the polys this shape needs, and loads any plaintexts it touches — so the
        // capture that follows sees a fully warm, allocation-free path. Costs one extra bootstrap
        // per cached shape, once.
        {
            Ciphertext warm_ct(cc_);
            warm_ct.copy(ctxt);
            BootstrapEager(warm_ct, slots, prescaled);
            cudaStreamSynchronize(warm_ct.c0.GPU[0].getS().ptr());
        }

        const double entry_nf = ctxt.NoiseFactor;
        e.slot = std::make_unique<Ciphertext>(cc_);
        e.slot->copy(ctxt);                       // full copy: structure is built HERE, pre-capture
        e.in_level = ctxt.getLevel();
        e.in_noise = ctxt.NoiseLevel;
        e.in_slots = ctxt.slots;
        // Record the entry buffers while the slot still has its ENTRY structure.
        e.in_bufs.clear();
        e.slot->c0.appendLiveLimbBuffers(e.in_bufs);
        e.slot->c1.appendLiveLimbBuffers(e.in_bufs);

        cudaStream_t s = e.slot->c0.GPU[0].getS().ptr();
        static cudaEvent_t cev = [] {
            cudaEvent_t ev = nullptr;
            cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
            return ev;
        }();
        std::vector<cudaStream_t> forked;
        cudaGraph_t g = nullptr;
        // Exclude the residency worker for the duration of the capture, then drain everything it
        // already submitted so no pre-capture event is left outstanding (both legal here: we are
        // not capturing yet).
        FIDESlib::captureGateLockExclusive();
        struct GateRelease {
            ~GateRelease() { FIDESlib::captureGateUnlockExclusive(); }
        } _capture_gate;
        cudaDeviceSynchronize();
        cudaGetLastError();
        if (cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed) != cudaSuccess) {
            e.poisoned = true;
            cudaGetLastError();
            return false;
        }
        FIDESlib::captureParamReset();
        g_bts_raise_param = BtsRaiseParam{};
        g_bts_restore_level = -1;
        FIDESlib::captureForkAll(s, forked, cev);
        BootstrapEager(*e.slot, slots, prescaled);
        FIDESlib::captureJoinAll(s, forked, cev);
        const cudaError_t ee = cudaStreamEndCapture(s, &g);
        if (ee != cudaSuccess || g == nullptr) {
            std::fprintf(stderr, "[bts_cache] capture failed (%s) shape lvl=%d slots=%d — eager for this shape\n",
                         cudaGetErrorName(ee), key.level, slots);
            if (g) cudaGraphDestroy(g);
            e.poisoned = true;
            cudaGetLastError();
            return false;
        }
        const cudaError_t ei = cudaGraphInstantiateWithFlags(&e.exec, g, 0);
        cudaGraphDestroy(g);
        if (ei != cudaSuccess || e.exec == nullptr) {
            std::fprintf(stderr, "[bts_cache] instantiate failed (%s) — eager for this shape\n",
                         cudaGetErrorName(ei));
            e.exec = nullptr;
            e.poisoned = true;
            cudaGetLastError();
            return false;
        }

        // The host code RAN during capture, so the slot now carries the post-bootstrap metadata and
        // the final limb structure. Both are the invariants every later replay restores/checks.
        // (The pool/workspace swap is undone by _ws_guard on every exit path.)
        e.out_level = e.slot->getLevel();
        e.out_noise = e.slot->NoiseLevel;
        e.out_slots = e.slot->slots;
        e.out_nf = e.slot->NoiseFactor;
        e.fingerprint = fingerprintOf(*e.slot);
        // Harvest the tagged parameter slots. If either is missing the graph is NOT reusable
        // across correction factors, so pin this entry to the factor it was captured with.
        e.raise_slots = FIDESlib::captureParamSlots(kBtsParamRaise);
        e.restore_slots = FIDESlib::captureParamSlots(kBtsParamRestore);
        e.raise_param = g_bts_raise_param;
        e.cap_correction = g_bts_raise_param.correction;
        e.cap_eff_cf = effCorrectionFactor(cc, slots);
        e.cap_prescale = cc.getBtsPreScale();
        e.cap_entry_nf = entry_nf;
        e.cap_restore_level = g_bts_restore_level;
        FIDESlib::captureParamReset();

        if (cudaGraphLaunch(e.exec, s) != cudaSuccess || cudaStreamSynchronize(s) != cudaSuccess) {
            std::fprintf(stderr, "[bts_cache] first launch failed — eager for this shape\n");
            cudaGraphExecDestroy(e.exec);
            e.exec = nullptr;
            e.poisoned = true;
            cudaGetLastError();
            return false;
        }
        std::fprintf(stderr,
                     "[bts_cache] captured shape #%zu: slots_arg=%d lvl=%d->%d noise=%d->%d ct_slots=%d->%d"
                     " nf=%.17g aux_owned=%zu raise_slots=%zu restore_slots=%zu raise_valid=%d\n",
                     bts_cache.size(), slots, e.in_level, e.out_level, e.in_noise, e.out_noise, e.in_slots,
                     e.out_slots, ctxt.NoiseFactor, e.owned_aux.size(), e.raise_slots.size(),
                     e.restore_slots.size(), (int)e.raise_param.valid);
        ctxt.copy(*e.slot);
        cudaGetLastError();
        return true;
    }

    // ── HIT ────────────────────────────────────────────────────────────────────────────────
    // ⚠️ THE GUARD THAT MAKES THIS SAFE TO SHIP. add.72's hardest lesson was that a graph can run
    // with sync=cudaSuccess and sanitizer-clean and still compute garbage (err_max 4.29e+132) when
    // an address it baked has moved. Anything that reallocates a slot limb makes every kernel node
    // address the wrong buffer, silently. So verify the addresses before trusting the exec.
    if (fingerprintOf(*e.slot) != e.fingerprint) {
        std::fprintf(stderr, "[bts_cache] FINGERPRINT MOVED (lvl=%d slots=%d) — dropping exec, going eager\n",
                     key.level, slots);
        cudaGraphExecDestroy(e.exec);
        e.exec = nullptr;
        e.poisoned = true;
        return false;
    }

    // Copy-in: raw device-to-device into the ENTRY buffers. Deliberately NOT RNSPoly::copy or even
    // copyShallow — both touch the slot's structure, and the slot must stay frozen in the shape the
    // graph was recorded against. The caller's live limbs are laid out in the same order.
    std::vector<std::pair<void*, size_t>> src;
    ctxt.c0.appendLiveLimbBuffers(src);
    ctxt.c1.appendLiveLimbBuffers(src);
    if (src.size() != e.in_bufs.size()) {
        std::fprintf(stderr, "[bts_cache] input limb count %zu != captured %zu — eager\n", src.size(),
                     e.in_bufs.size());
        cudaGraphExecDestroy(e.exec);
        e.exec = nullptr;
        e.poisoned = true;
        return false;
    }
    // ⚠️ btsPreScale is NOT in the key and is NOT parameterised — deliberately.
    // It folds into `constantEvalMult` (:198), a multiply that runs unconditionally, so it is not
    // a node that appears or disappears: it COULD be re-pointed per replay exactly like the
    // correction factor. It is not, because every plan on these arms runs prescale 1.0 (the CF
    // range covers what a prescale would buy), so the machinery would serve a case that never
    // occurs. What must not happen is silently reusing a graph captured at a different value —
    // the same wrong-answer class as the correction factor — hence an explicit guard rather than
    // an assumption. If it ever fires, take the eager path and say so.
    if (cc.getBtsPreScale() != e.cap_prescale) {
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1) < 3)
            std::fprintf(stderr, "[bts_cache] prescale %.17g != captured %.17g — eager for this call\n",
                         cc.getBtsPreScale(), e.cap_prescale);
        return false;
    }

    // ⭐ RE-POINT THE GRAPH AT THIS CALL'S CORRECTION FACTOR (add.166 add.73).
    // The graph's memcpy nodes copy these scalars out of the pinned arena on every replay, so
    // rewriting the arena is enough — no recapture, and the factor stays out of the cache key.
    // The level is fixed by the shape, so the raise scalar just rescales by 2^(cap - now).
    // ⚠️ The arena slots are read by the graph's memcpy NODES during execution, and replay is
    // launched asynchronously. Overwriting them from the host while a previous replay of this same
    // exec is still in flight corrupts that run's scalars — which is exactly when KL jumped from
    // 0.78 to 62 (and then stayed pinned at saturation regardless of what value was written).
    // Drain this slot's stream before touching the arena. If this is the cause, the fix is real
    // but costs a sync per hit; a per-slot double buffer would remove it again.
    if (!e.raise_slots.empty() || !e.restore_slots.empty())
        cudaStreamSynchronize(e.slot->c0.GPU[0].getS().ptr());

    if (!e.raise_slots.empty() && e.raise_param.valid) {
        // correction = effCorrectionFactor(slots) - deg, and `deg` is a property of the SHAPE the
        // key already pins, so the delta in `correction` is exactly the delta in the effective
        // factor — no need to re-derive deg.
        const uint32_t now_corr =
            (uint32_t)((int)e.cap_correction + (int)effCorrectionFactor(cc, slots) - (int)e.cap_eff_cf);
        // Rebuild the raise scalar from ModRaise's own formula.
        // ⚠️ ModRaise reads NoiseFactor AFTER the pending-rescale drain, not at bootstrap entry —
        // using the entry value scaled the whole result to ~0 (z_mass 0.0000, KL 0.78/3.16/2.32,
        // bit-identical run to run, i.e. deterministic and not a race). The drain is determined by
        // the shape, which the key pins, so its effect is the fixed ratio captured here.
        const double drain_ratio =
            (e.cap_entry_nf != 0.0) ? (e.raise_param.sourceSF / e.cap_entry_nf) : 1.0;
        const double nf = ctxt.NoiseFactor * drain_ratio;
        const double adj = (e.raise_param.targetSF / nf) * (e.raise_param.modToDrop / nf) *
                           std::pow(2.0, -(double)now_corr);
        const std::vector<uint64_t> res = cc.ElemForEvalMult(e.raise_param.level, adj);
        for (auto& sl : e.raise_slots)
            std::memcpy(sl.first, res.data(), std::min(sl.second, res.size() * sizeof(uint64_t)));
        if (e.cap_restore_level >= 0) {
            const uint64_t corFactor = (uint64_t)1 << now_corr;
            std::vector<uint64_t> op_(e.cap_restore_level + 1);   // NOT the raise level
            for (size_t i = 0; i < op_.size(); ++i)
                op_[i] = corFactor % cc.prime[i].p;
            for (auto& sl : e.restore_slots)
                std::memcpy(sl.first, op_.data(), std::min(sl.second, op_.size() * sizeof(uint64_t)));
        }
    }

    // ⚠️ ORDER THE COPY-IN AFTER THE PREVIOUS HIT'S COPY-OUT (add.166 add.73).
    // The slot is reused by every call of this shape. Copy-out reads it on the CALLER's streams
    // while copy-in writes it on the SLOT's stream, and nothing ordered those: a later bootstrap
    // could overwrite the slot while an earlier token's result was still being read out. The
    // damage compounds with reuse, which is the KL 0.78 -> 3.16 -> 2.32 trajectory across tokens.
    for (auto& g : ctxt.c0.GPU)
        e.slot->c0.GPU[0].s.wait(g.s);
    for (auto& g : ctxt.c1.GPU)
        e.slot->c0.GPU[0].s.wait(g.s);

    cudaStream_t s = e.slot->c0.GPU[0].getS().ptr();
    for (size_t i = 0; i < src.size(); ++i) {
        if (src[i].second != e.in_bufs[i].second) {
            std::fprintf(stderr, "[bts_cache] input limb %zu width changed — eager\n", i);
            cudaGraphExecDestroy(e.exec);
            e.exec = nullptr;
            e.poisoned = true;
            return false;
        }
        cudaMemcpyAsync(e.in_bufs[i].first, src[i].first, src[i].second, cudaMemcpyDeviceToDevice, s);
    }

    // ⚠️ NO cudaStreamSynchronize HERE. The eager bootstrap returns asynchronously, so syncing per
    // replay adds a full pipeline drain the eager path never pays — measured +0.35 ms/bts, i.e. it
    // ate more than the dispatch gap the graph exists to remove. Ordering is already correct
    // without it: the graph runs on the slot's stream and the copy-out below goes through
    // LimbPartition::copyLimb, whose opening Stream::wait covers exactly that stream.
    if (cudaGraphLaunch(e.exec, s) != cudaSuccess) {
        std::fprintf(stderr, "[bts_cache] replay launch failed at hit %llu — dropping exec, going eager\n",
                     (unsigned long long)e.hits);
        cudaGraphExecDestroy(e.exec);
        e.exec = nullptr;
        e.poisoned = true;
        cudaGetLastError();
        return false;
    }
    ++e.hits;

    // ⚠️ ORDER THE COPY-OUT AGAINST THE GRAPH, WITHOUT A DEVICE SYNC.
    // A graph launched on stream `s` executes ALL its work as part of `s` — the streams that were
    // forked during capture became graph nodes, not real streams. So the slot's OTHER streams (c1's
    // in particular) carry no dependency on the replayed work, and copyLimb's opening Stream::wait
    // waits on those. That is why dropping the sync outright returned err_max 4.8e+132 while the
    // cache mechanically reported hits: the copy-out raced the graph.
    // Making them wait on the origin costs two events instead of a full pipeline drain. Note
    // Stream::wait re-records the source only when `updated` is false — `ptr()` cleared it above,
    // so the event this records is taken AFTER the launch, which is precisely the guarantee needed.
    // (Reusing a stale event is capture blocker #5's mechanism, add.69, in reverse.)
    {
        Stream& origin = e.slot->c0.GPU[0].s;
        for (auto& g : e.slot->c0.GPU)
            if (&g.s != &origin)
                g.s.wait(origin);
        for (auto& g : e.slot->c1.GPU)
            if (&g.s != &origin)
                g.s.wait(origin);
    }

    // Replay ran no host code, but the slot was left carrying the post-bootstrap metadata by the
    // capture and nothing since has changed it, so the delta is already applied. Re-assert it
    // anyway: it is free, and it documents that the output shape is a property of the SHAPE KEY.
    e.slot->NoiseLevel = e.out_noise;
    e.slot->NoiseFactor = e.out_nf;
    e.slot->slots = e.out_slots;

    ctxt.copy(*e.slot);
    // ⚠️ Report a RUNNING TOTAL, not the first few events. The previous print was capped at 3
    // lines, so `grep -c` reported "3 hits" no matter what and made the hit rate unreadable —
    // it looked like 18 captures against 3 hits when the real ratio was 18 against ~540.
    const uint64_t tot = ++g_cache_hits;
    if ((tot & 127u) == 0)
        std::fprintf(stderr, "[bts_cache] hits=%llu captures=%zu (last lvl=%d slots=%d)\n",
                     (unsigned long long)tot, bts_cache.size(), key.level, slots);
    cudaGetLastError();
    return true;
}

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
    if (corPre && !skipCorFactor() && corFactor != 1) {
        // BTS_PARAM_RESTORE (add.166 add.73). ⚠️ MUST be the pair inside BootstrapEager — the
        // function the graph actually captures. An identical pair in BootstrapCPUraise was tagged
        // first by mistake and never fired (restore_slots=0 on every shape), which sent three
        // successive fixes chasing the raise scalar that was already correct.
        FIDESlib::captureParamBegin(kBtsParamRestore);
        if (FIDESlib::captureActive())
            btsRecordRestoreLevel(ctxt.getLevel());
        multIntScalar(ctxt, corFactor);
        FIDESlib::captureParamEnd();
    }

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

    if (!corPre && !skipCorFactor() && corFactor != 1) {
        FIDESlib::captureParamBegin(kBtsParamRestore);
        if (FIDESlib::captureActive())
            btsRecordRestoreLevel(ctxt.getLevel());
        multIntScalar(ctxt, corFactor);
        FIDESlib::captureParamEnd();
    }
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
            FIDESlib::captureParamBegin(kBtsParamRaise);
            if (FIDESlib::captureActive())
                btsRecordRaiseParam(ctxt.getLevel(), adjustmentFactor, correction, targetSF, modToDrop, sourceSF);
            ctxt.multScalar(adjustmentFactor);
            FIDESlib::captureParamEnd();
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
