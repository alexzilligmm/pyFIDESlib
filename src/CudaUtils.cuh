//
// Created by carlosad on 14/03/24.
//

#ifndef FIDESLIB_CUDAUTILS_CUH
#define FIDESLIB_CUDAUTILS_CUH

#include <cstdlib>   // _Exit

//#define NCCL

#include <cuda_runtime.h>
#include <execinfo.h>
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace FIDESlib {

extern std::vector<cudaDeviceProp> GPUprop;
void initGPUprop();
int GetTargetThreads(int id);

enum NVTX_CATEGORIES { NONE, LIFETIME, FUNCTION };

void CudaNvtxStart(const std::string msg, NVTX_CATEGORIES cat = FUNCTION, int val = 0);
void CudaNvtxStop(const std::string msg = "", NVTX_CATEGORIES cat = FUNCTION);
class CudaNvtxRange {
    const std::string msg;
    const NVTX_CATEGORIES cat;
    bool valid = true;

   public:
    explicit CudaNvtxRange(const std::string msg, NVTX_CATEGORIES cat = FUNCTION, int val = 0) : msg(msg), cat(cat) {
        CudaNvtxStart(msg, cat, val);
    }

    CudaNvtxRange(CudaNvtxRange&& r) noexcept : msg(r.msg), cat(r.cat) {
        this->valid = r.valid;
        r.valid = false;
    }

    ~CudaNvtxRange() {
        if (valid)
            CudaNvtxStop(msg, cat);
    }
};

int getNumDevices();

void CudaHostSync();
inline void breakpoint() {}

/* FATAL CUDA ERROR EXIT (2026-08-04). These three macros used `exit(0)`, which was wrong twice
 * over: (1) it reports SUCCESS for a fatal CUDA error — the reason CLAUDE.md has to say "exit
 * codes are NOT evidence" and gate on a PASS marker instead; and (2) exit() runs atexit hooks
 * and static destructors, which deadlock against a CUDA context that has just died, leaving the
 * process ALIVE and holding its whole device allocation (~40 GB of rotation keys here) until
 * killed by PID. On a one-GPU box that silently blocks the next run.
 * `_Exit(1)` skips all of that: the kernel reclaims the device memory immediately, and the
 * status is finally nonzero. Diagnostics are unaffected — the backtrace and message are already
 * printed above. */
#define CudaCheckErrorMod                                                                    \
    do {                                                                                     \
        cudaDeviceSynchronize();                                                             \
        cudaError_t e = cudaGetLastError();                                                  \
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {                    \
                                                                                             \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            FIDESlib::breakpoint();                                                          \
            _Exit(1); /* NOT exit(0): see note above */                                                                         \
        }                                                                                    \
    } while (0)

#define CudaCheckErrorModMGPU                                                                \
    do {                                                                                     \
        cudaStreamSynchronize(0);                                                            \
        cudaError_t e = cudaGetLastError();                                                  \
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {                    \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            FIDESlib::breakpoint();                                                          \
            _Exit(1); /* NOT exit(0): see note above */                                                                         \
        }                                                                                    \
    } while (0)

#define CudaCheckErrorModNoSync                                                                                   \
    do {                                                                                                          \
        /*cudaDeviceSynchronize();*/                                                                              \
        cudaError_t e = cudaGetLastError();                                                                       \
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled && e != cudaErrorGraphExecUpdateFailure) { \
            void* array[10];                                                                                      \
            size_t size;                                                                                          \
            size = backtrace(array, 10);                                                                          \
            backtrace_symbols_fd(array, size, STDERR_FILENO);                                                     \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));                      \
            FIDESlib::breakpoint();                                                                               \
            _Exit(1); /* NOT exit(0): see note above */                                                                                              \
        }                                                                                                         \
    } while (0)

#define NCCLCHECK(cmd)                                                                              \
    do {                                                                                            \
        ncclResult_t res = cmd;                                                                     \
        if (res != ncclSuccess) {                                                                   \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
            exit(EXIT_FAILURE);                                                                     \
        }                                                                                           \
    } while (0)

class Event;

extern std::map<void*, int> free;

class Stream {
   private:
    cudaStream_t ptr_ = nullptr;

   public:
    cudaEvent_t ev = nullptr;
    bool updated = false;
    //Event ev;

    void init(int priority = 0);

    cudaStream_t ptr() {
        updated = false;
        return ptr_;
    }

    void initDefault();

    //void wait(const Event &ev) const;
    void wait(Stream& s, bool external = false);
    void wait(cudaStream_t s);

    Stream();

    Stream(Stream& s) = delete;

    Stream(const Stream& s) = delete;

    Stream& operator=(const Stream&) = delete;

    Stream(Stream&& s) noexcept;

    ~Stream();

    void record(bool external = false);

    void wait_recorded(const Stream& s);

    void capture_begin();

    void capture_end();
};

template <bool capture>
void run_in_graph(cudaGraphExec_t& exec, Stream& s, std::function<void()> run);

void* GPUmalloc(int id, int bytes, cudaStream_t stream, bool cache = false);
void GPUfree(void* ptr, int id, int bytes, cudaStream_t stream, bool cache = false);

/* KSK L2 persisting-window probe (Blackwell lever #1, HANDOFF_blackwell_levers.md).
 * setPersistingL2Window raises cudaLimitPersistingL2CacheSize and installs an
 * accessPolicyWindow(hitProp=persisting) over [base, base+bytes) on EVERY FIDESlib stream —
 * retroactively via a registry of live streams, and on each stream created afterwards
 * (Stream::init applies the stored window). One window per process; a second call replaces
 * it. Streams outside FIDESlib (none launch KSK readers) are unaffected. The window is
 * read-path metadata only: no kernel, value or schedule changes, bit-exact by construction. */
void setPersistingL2Window(void* base, size_t bytes);
namespace detail {
void registerL2WindowStream(cudaStream_t s);    // Stream::init only
void unregisterL2WindowStream(cudaStream_t s);  // Stream teardown only
}  // namespace detail

}  // namespace FIDESlib
#endif  //FIDESLIB_CUDAUTILS_CUH
