//
// Created by carlosad on 25/03/24.
//

#include <atomic>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <cassert>
#include <list>
#include <mutex>
#include <set>
#include <string>
#include <shared_mutex>
#include <thread>
#include <unordered_set>
#include "CudaUtils.cuh"

#include <iostream>
#include <cub/detail/nvtx3.hpp>

//#include "driver_types.h"
#include "CKKS/Context.cuh"
#define DISABLE_STREAMS false

namespace FIDESlib {

struct my_domain {
    static constexpr char const* name{"FIDESlib"};
};

nvtx3::domain const& D = nvtx3::domain::get<my_domain>();

std::map<std::string, std::pair<std::unique_ptr<nvtx3::unique_range_in<my_domain>>, int>> lifetimes_map;

/* FIDESLIB_NVTX (default 0): NVTX ranges are a profiling aid, but they were also the last
 * un-audited SHARED-STATE writer on the two-ct path — the LIFETIME category mutates the
 * unguarded `lifetimes_map` std::map on every Ciphertext construction/destruction, so two
 * host threads corrupt the map (S7/FAILURE 2.33 class). Gated off by default; opt in with
 * FIDESLIB_NVTX=1 for single-threaded nsys runs. (Ported from rational32, where the same
 * gate measured wall-NEUTRAL — this is a correctness/hygiene gate, not a perf lever.) */
bool cudaNvtxEnabled() {
    static const bool v = [] {
        const char* e = std::getenv("FIDESLIB_NVTX");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return v;
}

void CudaNvtxStart(const std::string msg, NVTX_CATEGORIES cat, int val) {
    if (!cudaNvtxEnabled())
        return;

    if (cat == FUNCTION) {
        using namespace nvtx3;
        int size = msg.size();
        const event_attributes attr{msg,
                                    rgb{(uint8_t)(255 - 101 * msg[size / 6]), (uint8_t)(255 - 101 * msg[size * 3 / 6]),
                                        (uint8_t)(255 - 101 * msg[size * 5 / 6])},
                                    payload{val}, category{static_cast<unsigned int>(cat)}};

        nvtxDomainRangePushEx_impl_init_v3(D, reinterpret_cast<const nvtxEventAttributes_t*>(&attr));
        //nvtxRangePushEx(reinterpret_cast<const nvtxEventAttributes_t*>(&attr));
    } else if (cat == LIFETIME) {

        using namespace nvtx3;
        int size = msg.size();
        auto& [r, i] = lifetimes_map[msg];
        std::string m = std::to_string(i + 1) + std::string(" x ") + msg;
        const event_attributes attr{m,
                                    rgb{(uint8_t)(255 - 101 * msg[size / 6]), (uint8_t)(255 - 101 * msg[size * 3 / 6]),
                                        (uint8_t)(255 - 101 * msg[size * 5 / 6])},
                                    payload{i + 1}, category{static_cast<unsigned int>(cat)}};
        i = i + 1;
        if (!r) {
            r = std::make_unique<unique_range_in<my_domain>>(attr);
        } else {
            *r = unique_range_in<my_domain>(attr);
        }
    }
    //nvtxRangePushA(msg.c_str());
}

void CudaNvtxStop(const std::string msg, NVTX_CATEGORIES cat) {
    if (!cudaNvtxEnabled())
        return;
    if (cat == FUNCTION) {
        nvtxDomainRangePop(D);
    } else if (cat == LIFETIME) {
        using namespace nvtx3;
        int size = msg.size();

        auto& [r, i] = lifetimes_map[msg];
        std::string m = std::to_string(i - 1) + std::string(" x ") + msg;
        const event_attributes attr{m,
                                    rgb{(uint8_t)(255 - 101 * msg[size / 6]), (uint8_t)(255 - 101 * msg[size * 3 / 6]),
                                        (uint8_t)(255 - 101 * msg[size * 5 / 6])},
                                    payload{i - 1}, category{static_cast<unsigned int>(cat)}};

        i = i - 1;
        if (i <= 0) {
            if (r) {
                r.reset();
            }
        } else {
            *r = unique_range_in<my_domain>(attr);
        }

        //nvtxRangePushEx(reinterpret_cast<const nvtxEventAttributes_t*>(&attr));
    }
}

int getNumDevices() {
    int d;
    cudaGetDeviceCount(&d);
    return d;
};

void CudaHostSync() {
    cudaDeviceSynchronize();
}

template <bool capture>
void run_in_graph(cudaGraphExec_t& exec, Stream& s, std::function<void()> run) {
    cudaGraph_t graph;
    if constexpr (capture) {
        cudaStreamBeginCapture(s.ptr(), cudaStreamCaptureModeRelaxed);
        CudaCheckErrorModNoSync;
    }
    run();
    if constexpr (capture) {
        cudaStreamEndCapture(s.ptr(), &graph);
        if (!exec) {
            cudaGraphInstantiateWithFlags(&exec, graph, cudaGraphInstantiateFlagUseNodePriority);
            //cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
            CudaCheckErrorModNoSync;
        } else {
            if (cudaGraphExecUpdate(exec, graph, NULL) != cudaSuccess) {
                CudaCheckErrorModNoSync;
                // only instantiate a new graph if update fails
                cudaGraphExecDestroy(exec);
                cudaGraphInstantiateWithFlags(&exec, graph, cudaGraphInstantiateFlagUseNodePriority);
                //cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
                CudaCheckErrorModNoSync;
            }
        }
        cudaGraphDestroy(graph);
        cudaGraphLaunch(exec, s.ptr());
    }
}

template void run_in_graph<false>(cudaGraphExec_t& exec, Stream& s, std::function<void()> run);

template void run_in_graph<true>(cudaGraphExec_t& exec, Stream& s, std::function<void()> run);

/*
    void Stream::wait(const Event &ev) const {
        cudaStreamWaitEvent(ptr, ev.ptr());
    }
*/
void Stream::capture_begin() {
    CudaCheckErrorMod;
    std::cout << "Hello capture" << std::endl;
    cudaStreamCaptureStatus cap;
    cudaStreamIsCapturing(ptr(), &cap);

    CudaCheckErrorMod;
    if (cap == cudaStreamCaptureStatusNone) {
        std::cout << "None" << std::endl;
        cudaStreamBeginCapture(ptr(), cudaStreamCaptureModeGlobal);
    } else if (cap == cudaStreamCaptureStatusActive) {
        std::cout << "Fail: activo" << std::endl;
    } else if (cap == cudaStreamCaptureStatusInvalidated) {
        std::cout << "Fail: invalidado" << std::endl;
    } else {
        std::cout << "Fail" << std::endl;
    }
    CudaCheckErrorMod;
}

void Stream::capture_end() {
    cudaGraph_t graph;
    cudaStreamEndCapture(ptr(), &graph);
    CudaCheckErrorMod;
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    CudaCheckErrorMod;
    cudaGraphDestroy(graph);
    CudaCheckErrorMod;
    cudaGraphLaunch(graphExec, 0);
    cudaGraphExecDestroy(graphExec);

    cudaStreamSynchronize(0);
}

void Stream::record(bool external) {
    if (ptr_ == 0)
        return;
    CudaCheckErrorModNoSync;
#if !DISABLE_STREAMS
    //cudaEventDestroy(ev);
    //cudaEventCreate(&ev, cudaEventDisableTiming);
    assert(ptr_ != nullptr);
    assert(ev != nullptr);
    cudaEventRecordWithFlags(ev, ptr_, external ? cudaEventRecordExternal : cudaEventRecordDefault);
    updated = true;
    recorded_in_capture = captureActive();   // see Stream::recorded_in_capture (add.166 add.73)
#endif
}

void Stream::wait_recorded(const Stream& s) {
    if (ptr_ == 0 || s.ptr_ == 0 || ptr_ == s.ptr_)
        return;
#if !DISABLE_STREAMS
    assert(s.updated);
    cudaStreamWaitEvent(ptr_, s.ev);
    this->updated = false;
#endif
}

void Stream::wait(Stream& s, bool external) {

#if !DISABLE_STREAMS
    //CudaCheckErrorModNoSync;
    if (ptr_ == 0 || s.ptr_ == 0 || ptr_ == s.ptr_)
        return;
    assert(ptr_ != nullptr);
    assert(s.ptr_ != nullptr);
    assert(ev != nullptr);
    // ⚠️ WHILE CAPTURING, ALWAYS re-record (add.166 add.69). `updated` means "this stream has a
    // valid event", but an event recorded BEFORE cudaStreamBeginCapture is uncaptured work, and
    // waiting on it from a capturing stream is cudaErrorStreamCaptureIsolation — the 5th and
    // subtlest capture blocker, and invisible from the error site alone because the WAIT is what
    // fails while the offending RECORD happened earlier and elsewhere.
    // Re-record when there is no event, when capturing (add.69 blocker #5: an event recorded
    // BEFORE BeginCapture is uncaptured work), and ALSO when the last record was made during a
    // capture and we are no longer capturing (add.73): that record became a graph node, so the
    // event was never actually recorded on this stream and waiting on it is 'invalid argument'.
    const bool stale_from_capture = s.recorded_in_capture && !captureActive();
    if (!s.updated || captureActive() || stale_from_capture) {
        assert(!external);  // Has to be recorded in the origin graph
        CudaCheckErrorModNoSync;
        cudaEventRecordWithFlags(s.ev, s.ptr_, cudaEventRecordDefault);
        s.updated = true;
        s.recorded_in_capture = captureActive();
        CudaCheckErrorModNoSync;
    }
    CudaCheckErrorModNoSync;
    cudaStreamWaitEvent(ptr_, s.ev, external ? cudaEventWaitExternal : cudaEventWaitDefault);
    this->updated = false;
#endif
    CudaCheckErrorModNoSync;
}

void Stream::wait(cudaStream_t s) {

#if !DISABLE_STREAMS
    //CudaCheckErrorModNoSync;
    if (s == 0 || ptr_ == 0)
        return;
    assert(ptr_ != nullptr);
    assert(ev != nullptr);
    CudaCheckErrorModNoSync;
    cudaEventRecordWithFlags(ev, s, cudaEventRecordDefault);
    updated = false;
    CudaCheckErrorModNoSync;

    CudaCheckErrorModNoSync;
    cudaStreamWaitEvent(ptr_, ev, cudaEventWaitDefault);
#endif
    CudaCheckErrorModNoSync;
}

/* KSK L2 persisting-window probe — see CudaUtils.cuh. State + registry are process-global:
 * FIDESlib streams all funnel through Stream::init/teardown, so the registry is complete by
 * construction. hitRatio is derived once at set time (carveout / window) so the hardware
 * randomly persists at most a carveout's worth of the window instead of thrashing it. */
namespace {
/* ⚠️ INTENTIONALLY NEVER DESTROYED (2026-08-29) — static-destruction-order fiasco, the same class
 * add.166 add.56 root-caused for g_staged_shared and commit 8c15703 fixed the same way.
 *
 * These are namespace statics, but the FIDESlib Contexts that own every Stream live in ANOTHER
 * static (the `std::map<Parameters, shared_ptr<ContextData>>` context cache, a different TU).
 * Destruction order across TUs is unspecified, and in practice this registry went FIRST: at exit
 * `__run_exit_handlers` destroyed the context map -> ~ContextData -> ~vector<LimbPartition> ->
 * ~Stream -> unregisterL2WindowStream(), which then erased from an ALREADY-DESTROYED std::set and
 * segfaulted inside _Rb_tree_rebalance_for_erase. Locking the destroyed mutex is UB for the same
 * reason, so both leak deliberately.
 *
 * Leaking them is free: a std::set of stream handles and a mutex, released by the kernel at exit.
 * This was masked until 2026-08-29 by CudaCheckErrorMod's _Exit(1) firing on the preceding
 * `cudaErrorCudartUnloading` — the process died before it could reach the crash. */
std::mutex& l2win_mtx = *new std::mutex();
std::set<cudaStream_t>& l2win_streams = *new std::set<cudaStream_t>();
cudaAccessPolicyWindow l2win{};  // num_bytes == 0 <=> no window configured (POD, trivially destructible)

void l2winApply(cudaStream_t s) {  // call with l2win_mtx held, l2win.num_bytes > 0
    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow = l2win;
    cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &attr);
}
}  // namespace

void setPersistingL2Window(void* base, size_t bytes) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    if (prop.persistingL2CacheMaxSize <= 0 || bytes == 0) {
        std::cerr << "[ksk_l2] persisting L2 unsupported on device " << dev << " — window not set\n";
        return;
    }
    const size_t carve = (size_t)prop.persistingL2CacheMaxSize;
    const size_t window = std::min(bytes, (size_t)prop.accessPolicyMaxWindowSize);
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, carve);

    std::lock_guard<std::mutex> lock(l2win_mtx);
    l2win.base_ptr = base;
    l2win.num_bytes = window;
    l2win.hitRatio = std::min(1.0f, (float)((double)carve / (double)window));
    l2win.hitProp = cudaAccessPropertyPersisting;
    l2win.missProp = cudaAccessPropertyNormal;
    for (cudaStream_t s : l2win_streams)
        l2winApply(s);
    std::cerr << "[ksk_l2] window base=" << base << " bytes=" << (window >> 20) << " MB (of " << (bytes >> 20)
              << " MB requested), carveout=" << (carve >> 20) << " MB, hitRatio=" << l2win.hitRatio << ", applied to "
              << l2win_streams.size() << " live streams\n";
    CudaCheckErrorMod;
}

namespace detail {
void registerL2WindowStream(cudaStream_t s) {
    if (!s)
        return;
    std::lock_guard<std::mutex> lock(l2win_mtx);
    l2win_streams.insert(s);
    if (l2win.num_bytes)
        l2winApply(s);
}
void unregisterL2WindowStream(cudaStream_t s) {
    if (!s)
        return;
    std::lock_guard<std::mutex> lock(l2win_mtx);
    l2win_streams.erase(s);
}
}  // namespace detail

// ── GRAPH-CAPTURE FORK/JOIN (add.166 add.69) ────────────────────────────────────────────────
// A bootstrap touches ~10-40 partition streams. cudaStreamBeginCapture only captures the origin
// stream plus streams that are FORKED FROM IT while capturing; a capturing stream waiting on an
// event recorded by a NON-capturing stream is cudaErrorStreamCaptureIsolation — which is exactly
// how the dry capture died:
//   CudaUtils.cu:235 'dependency created on uncaptured work in another stream'  (Stream::wait)
// So every live FIDESlib stream must be pulled into the capture up front and joined back before
// EndCapture (an un-joined forked stream is cudaErrorStreamCaptureUnjoined).
//
// The registry is free: `l2win_streams` already holds every live stream by construction
// (Stream::init/~Stream register/unregister), and add.59 gave it a never-destroyed lifetime.
// ⚠️ Single-threaded bring-up only. Capture is a property of the STREAM, not the thread, so any
// other host thread touching a forked stream would have its work silently swept into the graph.
// A stream CREATED DURING the capture is not forked by captureForkAll (it did not exist yet), and
// its first cross-stream wait is then cudaErrorStreamCaptureIsolation. `generate()`/`initStream()`
// do create streams lazily mid-bootstrap, so Stream::init has to join the capture on the spot.
std::atomic<cudaStream_t> g_capture_origin{nullptr};
std::atomic<cudaEvent_t> g_capture_ev{nullptr};
// ⭐ Capture is scoped to the CAPTURING THREAD (add.166 add.73). It is a property of the STREAM,
// but every behavioural switch keyed on captureActive() — GPUmalloc's cudaMalloc branch, GPUfree's
// no-op, CAP_SRC's arena staging, Stream::wait's forced re-record — must apply ONLY to the thread
// building the graph. Under threaded inference the residency worker runs concurrently on its own
// streams; a global flag silently changed ITS allocator and staging behaviour too, and let
// captureAdoptStream drag worker-created streams into the graph.
std::atomic<std::thread::id> g_capture_tid{};

// Stage a small HOST buffer into a persistent PINNED arena and return the stable pointer
// (add.166 add.71). Graph capture records a memcpy's SOURCE ADDRESS; this library sources many
// of them from function-local std::vectors, which are dead by replay time — that is what made a
// well-formed, launchable graph die executing with cudaErrorIllegalAddress. Copying into an arena
// that outlives the graph makes those sources valid.
// ⚠️ Contents are frozen at capture. That is correct for a capture-then-replay-once proof, and it
// is exactly the "stale scalar" hazard for a CACHED graph replayed on new data — see add.61.
// ⚠️ Bump allocator, never reset: one capture's worth of staging (~64 KB) leaks per capture.
// Bring-up only, same disposition as the GPUfree capture guard.
namespace {
std::mutex stage_mtx;
uint8_t* stage_base = nullptr;
size_t stage_off = 0, stage_cap = 0;
}  // namespace

// Graph-parameter tagging (add.166 add.73) - see CudaUtils.cuh for why this exists.
namespace {
thread_local int g_param_tag = -1;
std::mutex param_mtx;
std::map<int, std::vector<std::pair<void*, size_t> > > param_slots;
}  // namespace

void captureParamBegin(int tag) { g_param_tag = tag; }
void captureParamEnd() { g_param_tag = -1; }

std::vector<std::pair<void*, size_t> > captureParamSlots(int tag) {
    std::lock_guard<std::mutex> lk(param_mtx);
    auto it = param_slots.find(tag);
    return it == param_slots.end() ? std::vector<std::pair<void*, size_t> >{} : it->second;
}

void captureParamReset() {
    std::lock_guard<std::mutex> lk(param_mtx);
    param_slots.clear();
}

const void* stageForCapture(const void* src, size_t bytes) {
    if (!src || bytes == 0) return src;
    std::lock_guard<std::mutex> lk(stage_mtx);
    if (stage_base == nullptr) {
        stage_cap = 64ull << 20;   // 64 MB of pinned staging; ~64 KB is used per captured bootstrap
        if (cudaHostAlloc((void**)&stage_base, stage_cap, cudaHostAllocPortable) != cudaSuccess) {
            stage_base = nullptr;
            stage_cap = 0;
            return src;            // refuse to fail the op over a staging buffer
        }
    }
    const size_t need = (bytes + 255) & ~(size_t)255;
    if (stage_off + need > stage_cap) return src;   // exhausted: fall back, capture will just fail
    uint8_t* dst = stage_base + stage_off;
    stage_off += need;
    std::memcpy(dst, src, bytes);
    if (g_param_tag >= 0) {   // this stage belongs to a tagged graph parameter
        std::lock_guard<std::mutex> pk(param_mtx);
        param_slots[g_param_tag].emplace_back((void*)dst, bytes);
    }
    return dst;
}

bool captureActive() {
    return g_capture_origin.load(std::memory_order_relaxed) != nullptr &&
           g_capture_tid.load(std::memory_order_relaxed) == std::this_thread::get_id();
}

/** True while ANY thread is capturing — for callers that must stand back rather than change
 *  behaviour (the residency worker's quiesce barrier). */
namespace {
std::shared_mutex g_capture_gate;
}  // namespace
void captureGateLockShared() { g_capture_gate.lock_shared(); }
void captureGateUnlockShared() { g_capture_gate.unlock_shared(); }
void captureGateLockExclusive() { g_capture_gate.lock(); }
void captureGateUnlockExclusive() { g_capture_gate.unlock(); }

bool captureInFlight() {
    return g_capture_origin.load(std::memory_order_relaxed) != nullptr;
}

// Blocks GPUmalloc handed out via plain cudaMalloc because a capture was in flight (add.72). The
// route has to be remembered: cudaFreeAsync on a cudaMalloc'd pointer is undefined, the same trap
// `bufferSPECIALcudaMalloc` already records for generateSpecialLimb's communication arm. Empty on
// every run that never captures, so the lookup in GPUfree is one uncontended empty-set check.
std::mutex capture_raw_mtx;
std::unordered_set<void*> capture_raw;
// Fast-path gate. The free paths must not touch the SET without the lock — reading
// unordered_set::empty() while another thread inserts is a data race — and on every run that never
// captures the answer is always "empty", so a relaxed atomic makes the common case one load.
std::atomic<size_t> capture_raw_count{0};

void captureSafeMallocAsync(void** ptr, size_t bytes, cudaStream_t stream) {
    if (!captureActive()) {
        cudaMallocAsync(ptr, bytes, stream);
        return;
    }
    *ptr = nullptr;
    if (cudaMalloc(ptr, bytes) != cudaSuccess || *ptr == nullptr) {
        cudaGetLastError();
        return;
    }
    std::lock_guard<std::mutex> g(capture_raw_mtx);
    if (capture_raw.insert(*ptr).second)
        capture_raw_count.store(capture_raw.size(), std::memory_order_relaxed);
}

void captureSafeFreeAsync(void* ptr, cudaStream_t stream) {
    if (!ptr)
        return;
    if (capture_raw_count.load(std::memory_order_relaxed) != 0) {
        std::lock_guard<std::mutex> g(capture_raw_mtx);
        auto it = capture_raw.find(ptr);
        if (it != capture_raw.end()) {
            // Do NOT free while the capture that recorded uses of this buffer is still in flight —
            // that is the add.70 pool-aliasing hazard. Leave it registered; GPUfree/teardown will
            // reclaim it once no capture is active.
            if (captureActive())
                return;
            capture_raw.erase(it);
            capture_raw_count.store(capture_raw.size(), std::memory_order_relaxed);
            cudaFree(ptr);
            return;
        }
    }
    cudaFreeAsync(ptr, stream);
}

void captureAdoptStream(cudaStream_t s) {
    cudaStream_t origin = g_capture_origin.load(std::memory_order_relaxed);
    cudaEvent_t ev = g_capture_ev.load(std::memory_order_relaxed);
    if (!origin || !ev || !s || s == origin) return;
    cudaEventRecord(ev, origin);       // legal: origin is the capturing stream
    cudaStreamWaitEvent(s, ev, 0);     // pulls the newcomer into the capture
}

void captureForkAll(cudaStream_t origin, std::vector<cudaStream_t>& forked, cudaEvent_t ev) {
    g_capture_tid.store(std::this_thread::get_id(), std::memory_order_relaxed);
    g_capture_origin.store(origin, std::memory_order_relaxed);
    g_capture_ev.store(ev, std::memory_order_relaxed);
    forked.clear();
    cudaEventRecord(ev, origin);
    std::lock_guard<std::mutex> lock(l2win_mtx);
    for (cudaStream_t s : l2win_streams) {
        if (s == origin) continue;
        if (cudaStreamWaitEvent(s, ev, 0) == cudaSuccess) forked.push_back(s);
    }
}
void captureJoinAll(cudaStream_t origin, const std::vector<cudaStream_t>& forked, cudaEvent_t ev) {
    g_capture_origin.store(nullptr, std::memory_order_relaxed);
    g_capture_tid.store(std::thread::id{}, std::memory_order_relaxed);
    g_capture_ev.store(nullptr, std::memory_order_relaxed);
    for (cudaStream_t s : forked) {
        cudaEventRecord(ev, s);
        cudaStreamWaitEvent(origin, ev, 0);
    }
}

int low = -1;
int high = -1;
void Stream::init(int priority) {
    if (ptr_) {
        //free[ptr]++;
        detail::unregisterL2WindowStream(ptr_);
        cudaEventDestroy(ev);
        cudaStreamDestroy(ptr_);
        ptr_ = nullptr;
        ev = nullptr;
    }

#if !DISABLE_STREAMS
    if (high == -1) {
        cudaDeviceGetStreamPriorityRange(&low, &high);
    }

    int prio = low + priority * ((high - low - 1)) / 100;
    cudaStreamCreateWithPriority(&ptr_, 0 /*cudaStreamNonBlocking*/, prio);
    //cudaStreamCreateWithFlags(&ptr, cudaStreamNonBlocking);
    detail::registerL2WindowStream(ptr_);
    captureAdoptStream(ptr_);   // no-op unless a graph capture is in flight (add.166 add.69)

    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    cudaEventCreate(&ev, cudaEventDisableTiming);
#else
    ptr = nullptr;
    ev = nullptr;
#endif
    //free[ptr] = 0;
}

void Stream::initDefault() {
    ptr_ = 0;
    ev = nullptr;
    updated = true;
}

//std::map<void *, int> free;

Stream::~Stream() {
    if (ptr_) {
        //free[ptr]++;
        detail::unregisterL2WindowStream(ptr_);
        cudaStreamDestroy(ptr_);
        ptr_ = nullptr;
    }
    if (ev) {
        cudaEventDestroy(ev);
        ev = nullptr;
    }
}

Stream::Stream() = default;

Stream::Stream(Stream&& s) noexcept : ptr_(s.ptr_), ev(s.ev) {
    s.ptr_ = nullptr;
    s.ev = nullptr;
}

std::vector<cudaDeviceProp> GPUprop;

void initGPUprop() {
    if (GPUprop.empty()) {
        int count = 0;
        cudaGetDeviceCount(&count);
        for (int i = 0; i < count; ++i) {
            GPUprop.emplace_back();
            cudaGetDeviceProperties(&GPUprop.back(), i);

            std::cout << "GPU " << i << ": " << GPUprop[i].name << "\n SMs: " << GPUprop[i].multiProcessorCount
                      << ", SharedMem: " << GPUprop[i].sharedMemPerMultiprocessor / 1024l
                      << " KB, Blocks/SM: " << GPUprop[i].maxBlocksPerMultiProcessor
                      << ", Threads/SM: " << GPUprop[i].maxThreadsPerMultiProcessor
                      << ", L2 size: " << GPUprop[i].l2CacheSize / (1024l * 1024l)
                      << " MB, Bus: " << (long long)GPUprop[i].memoryBusWidth

                      << "-bit" << std::endl;
        }
    }
}

/* ⚠️ INTENTIONALLY NEVER DESTROYED (2026-08-29) — the SAME static-destruction-order fiasco as
 * l2win_streams above and as add.166 add.56 / commit 8c15703. `new T[8]` yields a plain T*, so
 * every `mempool_lock[id]` / `size_to_memory[id]` / `s[id]` use site is unchanged.
 *
 * PROVEN BY ASan (recipe: add.166 add.56 — setarch -R first, LD_PRELOAD scoped to the target):
 *   heap-use-after-free, READ of size 32768 in memmove
 *     #1 FIDESlib::GPUfree            <- push_back into the pool's free-list vector
 *     #7 FIDESlib::CKKS::ContextData::~ContextData
 *     #10 std::map<Parameters, shared_ptr<ContextData>>::~map   <- the static context cache
 *     #11 __run_exit_handlers
 *   freed by:  std::_Rb_tree<int, vector<void*>>::_M_erase  <- size_to_memory's OWN destructor
 *              #2 __GI_exit
 *   allocated by: FIDESlib::GPUmalloc
 * i.e. the pool map is destroyed by the exit handlers FIRST, and the context cache's destructor
 * then calls GPUfree, which push_backs into the destroyed free-list. Order across TUs is
 * unspecified, so the only robust fix is to outlive every possible caller.
 *
 * Cost: one 8-element pool per process, reclaimed by the kernel at exit; the DEVICE memory it
 * tracks is reclaimed by the driver regardless. Do not "fix the leak" — it is load-bearing. */
std::mutex* const mempool_lock = new std::mutex[8];

std::map<int, std::vector<void*>>* const size_to_memory = new std::map<int, std::vector<void*>>[8];

FIDESlib::Stream* const s = new FIDESlib::Stream[8];
//void* GPUmalloc(int id, int bytes, cudaStream_t stream, FIDESlib::CKKS::Context& cc) {
void* GPUmalloc(int id, int bytes, cudaStream_t stream, bool cache) {
    void* ptr = nullptr;

    uint64_t MBs = 1024;

    if (bytes < 64 * 1024) {
        int next_pow2 = 1024;
        while (next_pow2 < bytes) {
            next_pow2 *= 2;
        }
        bytes = next_pow2;
        cache = true;
        MBs = bytes / 1024;
    }

    if (cache && (bytes & (bytes - 1)) == 0) {
        // S7 thread-safety (2026-08-03): the WHOLE pooled path holds the lock — the map
        // operator[] (node insert), the empty-check/refill/pop sequence and the shared
        // per-id event were all racy under concurrent host threads. Cold path; the lock
        // is nanoseconds against a 28 ms bootstrap.
        std::lock_guard<std::mutex> guard(mempool_lock[id]);
        std::vector<void*>& free_limb = size_to_memory[id][bytes];

        if (s[id].ptr() == nullptr) {
            s[id].init();
        }
        CudaCheckErrorModNoSync;
        if (free_limb.empty()) {
            uint64_t* base;
            // Same rule as the unpooled fallback below (add.72): a slab refilled DURING a capture
            // must not be a graph memory node. Slab bases are never freed (only the carved chunks
            // are recycled), so plain cudaMalloc needs no route bookkeeping here.
            if (captureActive())
                cudaMalloc(&base, MBs * 1024 * 1024);
            else
                cudaMallocAsync(&base, MBs * 1024 * 1024, s[id].ptr());

            for (int i = 0; i < MBs * 1024 * 1024; i += bytes) {
                free_limb.emplace_back(((char*)base) + i);
            }
        }
        CudaCheckErrorModNoSync;

        if (stream != nullptr) {
            s[id].record();
            CudaCheckErrorModNoSync;
            cudaStreamWaitEvent(stream, s[id].ev);
        }

        CudaCheckErrorModNoSync;
        ptr = free_limb.back();
        free_limb.pop_back();
        //ptr = free_limb.front();
        //free_limb.pop_front();
        //std::cout << "get " << ptr << std::endl;
        return ptr;
    }

    //std::cout << bytes << std::endl;
    // ⚠️ CAPTURE BLOCKER #1 (add.166 add.69). This unpooled fallback allocates on the LEGACY
    // STREAM (`0`). FIDESlib streams are created BLOCKING (CudaUtils.cu:349 passes flags=0 with
    // cudaStreamNonBlocking commented out), so a legacy-stream op while one of them is capturing
    // is illegal: the dry capture died with
    //   LimbPartition.cu:581 'operation would make the legacy stream depend on a capturing
    //   blocking stream'  (= cudaErrorStreamCaptureImplicit)
    // Note WHY that site reaches the fallback at all: `bufferSPECIALbytes = N*meta*2*8` is not a
    // power of two, and GPUmalloc's pool is gated on power-of-two sizes (:437) — the same gate
    // that made add.64's slab a wash. Using the CALLER's stream is strictly more correct anyway
    // (the memory is used on that stream; the legacy version only worked by leaning on the
    // implicit sync that blocking streams give). Scoped to graph bring-up for now so the shipped
    // allocator path is byte-identical until capture is proven end to end.
    static const bool graph_bringup = [] {
        const char* e = std::getenv("FIDESLIB_BTS_GRAPH");
        return e && *e && std::atoi(e) != 0;
    }();
    if (0) {
        cudaMalloc(&ptr, bytes);
    } else if (captureActive()) {
        // ⚠️ WHILE CAPTURING, ALLOCATE OUTSIDE THE GRAPH (add.166 add.72). A cudaMallocAsync on a
        // capturing stream becomes a graph MEMORY NODE, whose buffer is owned by the graph and is
        // not valid at the moment replay's kernels write it — compute-sanitizer says exactly that:
        //   Invalid __global__ write ... dotProductLtBatchedPt3___
        //   Access to 0x… is potentially made before memory is allocated
        //   and is inside the nearest allocation at 0x… of size 9,437,184 bytes
        // 9,437,184 = N * |SPECIALmeta| * 2 * 8 = bufferSPECIAL (LimbPartition.cu:583), which
        // reaches this fallback because that size is not a power of two.
        // cudaMalloc is a HOST call, so it is not recorded, and the buffer simply outlives the
        // graph — the same trick LimbPartition::scratchGet uses. Remember the pointer so GPUfree
        // routes it to cudaFree: a cudaMalloc'd block must never reach cudaFreeAsync.
        cudaMalloc(&ptr, bytes);
        if (ptr) {
            std::lock_guard<std::mutex> g(capture_raw_mtx);
            if (capture_raw.insert(ptr).second)
                capture_raw_count.store(capture_raw.size(), std::memory_order_relaxed);
        }
    } else if (graph_bringup) {
        cudaMallocAsync(&ptr, bytes, stream);
    } else if (1) {
        cudaMallocAsync(&ptr, bytes, 0);
    } else {
        if (size_to_memory[id][bytes].empty()) {
            cudaSetDevice(id);
            cudaMallocAsync(&ptr, bytes, stream);
        } else {
            mempool_lock[id].lock();
            if (size_to_memory[id][bytes].empty()) {
                mempool_lock[id].unlock();
                cudaSetDevice(id);
                cudaMallocAsync(&ptr, bytes, stream);
            } else {
                ptr = size_to_memory[id][bytes].back();
                size_to_memory[id][bytes].pop_back();
                mempool_lock[id].unlock();
            }
        }
    }

    return ptr;
}

struct pointerdata {
    void* pointer;
    int id;
    int bytes;
};

void CUDART_CB streamCallback(void* userData) {

    auto* p = reinterpret_cast<pointerdata*>(userData);

    mempool_lock[p->id].lock();
    size_to_memory[p->id][p->bytes].push_back(p->pointer);
    mempool_lock[p->id].unlock();
    delete p;
}

/** FAILURE §7 ablation. `FIDESLIB_GPUFREE_SIZED=0` restores the pre-fix behaviour — every
 *  caller's byte count discarded, so a large block lands in the 1 KB bucket and is stranded.
 *  It exists because this is the SHARED allocator: the fix has to be A/B-able on the classic
 *  bootstrap, not just on the RR path that found it. Default ON (i.e. sized, correct). */
static bool gpufreeSized() {
    static const bool sized = [] {
        const char* e = std::getenv("FIDESLIB_GPUFREE_SIZED");
        return e == nullptr || std::atoi(e) != 0;
    }();
    return sized;
}

void GPUfree(void* ptr, int id, int bytes, cudaStream_t stream, bool cache) {
    // ⚠️ WHILE CAPTURING, DO NOT RECYCLE (add.166 add.70). Host code RUNS during capture, so the
    // bootstrap's temporaries (aux, the Chebyshev scratch) are destructed and would return their
    // buffers to the pool — but the recorded graph still writes to those exact addresses, so by
    // replay time the pool may have handed them to someone else. That is the pool-aliasing hazard
    // the add.61 design review rated Critical, and it is what made the first replay attempt die
    // with 'an illegal memory access was encountered' at Stream::wait.
    // Leaking for the duration of a capture is the minimal correct answer for the bring-up proof.
    // ⚠️ It IS a leak: every captured bootstrap keeps its temporaries forever. Acceptable while
    // proving correctness on a handful of bootstraps; a per-graph arena released with the exec is
    // the real fix before this is ever cached or shipped.
    if (captureActive())
        return;

    // A block GPUmalloc took from plain cudaMalloc during a capture (add.72) must go back the same
    // way, and must NOT enter the async pool — the graph that recorded it may still be replayed.
    if (capture_raw_count.load(std::memory_order_relaxed) != 0) {
        std::lock_guard<std::mutex> g(capture_raw_mtx);
        auto it = capture_raw.find(ptr);
        if (it != capture_raw.end()) {
            capture_raw.erase(it);
            capture_raw_count.store(capture_raw.size(), std::memory_order_relaxed);
            cudaFree(ptr);
            return;
        }
    }

    uint64_t MBs = 1024;
    if (!gpufreeSized())
        bytes = 0;

    if (bytes < 64 * 1024) {
        int next_pow2 = 1024;
        while (next_pow2 < bytes) {
            next_pow2 *= 2;
        }
        bytes = next_pow2;
        cache = true;
        MBs = bytes / 1024;
    }

    if (cache && (bytes & (bytes - 1)) == 0) {
        // S7 thread-safety: same full-lock rule as GPUmalloc (map insert + shared event).
        std::lock_guard<std::mutex> guard(mempool_lock[id]);
        if (s[id].ptr() == nullptr) {
            s[id].init();
        }
        std::vector<void*>& free_limb = size_to_memory[id][bytes];

        CudaCheckErrorModNoSync;
        // cudaDeviceSynchronize();
        if (stream != nullptr) {
            s[id].wait(stream);
        }
        CudaCheckErrorModNoSync;
        free_limb.emplace_back(ptr);
        //std::cout << "free " << ptr << std::endl;
        return;
    }
    if (s[id].ptr() == nullptr) {
        s[id].init();
    }

    if (0) {
        cudaFree(ptr);
    } else if (1) {
        cudaFreeAsync(ptr, stream);
    } else {
        auto* p = new pointerdata;
        p->id = id;
        p->bytes = bytes;
        p->pointer = ptr;
        cudaLaunchHostFunc(stream, streamCallback, p);
    }
}

int GetTargetThreads(int id) {
    return GPUprop[id].multiProcessorCount * GPUprop[id].maxThreadsPerMultiProcessor;
}

}  // namespace FIDESlib