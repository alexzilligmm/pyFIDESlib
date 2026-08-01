//
// Created by carlosad on 4/04/24.
//

#ifndef FIDESLIB_NTT_CUH
#define FIDESLIB_NTT_CUH

#include <cinttypes>
#include "CKKS/forwardDefs.cuh"
#include "ConstantsGPU.cuh"

namespace FIDESlib {

struct FusedIterationsParams {
    struct __align__(128) AtomicCounter {
        uint32_t n = 0;
        uint32_t pad[(128 - sizeof(uint32_t)) / sizeof(uint32_t)];
    };
    AtomicCounter counters[MAXP];
    struct Conf {
        dim3 grid;
        dim3 block;
    };
    Conf first;
    Conf second;
};

/* Utility function, no real use other than testing. */
template <typename T>
__global__ void Bit_Reverse(T* dat, uint32_t N);

/* Get pointer to kernel, needed for explicit Cuda Graph construction. */
void* get_NTT_reference(bool second);

// ------------------------------------- INTT ----------------------------------------
/** Kernel fusions */
enum INTT_MODE { INTT_NONE, INTT_MULT_AND_SAVE, INTT_MULT_AND_ACC, INTT_ROTATE_AND_SAVE, INTT_SQUARE_AND_SAVE };

template <typename T, bool second = true, ALGO algo = ALGO_SHOUP, INTT_MODE mode = INTT_NONE>
__global__ void INTT_(const Global::Globals* Globals, T* __restrict__ dat, const int __grid_constant__ primeid,
                      T* __restrict__ res, const T* __restrict__ dat2 = nullptr, T* __restrict__ res0 = nullptr,
                      T* __restrict__ res1 = nullptr, const T* __restrict__ kska = nullptr,
                      const T* __restrict__ kskb = nullptr, T* __restrict__ c0 = nullptr,
                      const T* __restrict__ c0tilde = nullptr);

template <bool second, ALGO algo, INTT_MODE mode>
__global__ void INTT_(const Global::Globals* Globals, void** __restrict__ dat, const int __grid_constant__ primeid_init,
                      void** __restrict__ res, void** __restrict__ dat2 = nullptr, void** __restrict__ res0 = nullptr,
                      void** __restrict__ res1 = nullptr, void** __restrict__ kska = nullptr,
                      void** __restrict__ kskb = nullptr, void** __restrict__ c0 = nullptr,
                      void** __restrict__ c0tilde = nullptr);

// ------------------------------------- NTT ----------------------------------------
/** Kernel fusions */
// NTT_RESCALE2 (n32 speed): fused composite DOUBLE prime drop — one wide pass instead of two
// sequential NTT_RESCALE passes. Sequential-drop semantics preserved exactly (bit-identical to
// two NTT_RESCALE passes): the second top's once-divided value w is derived per-coefficient in
// coeff domain, and per-prime constants commute with the (exact, modular) NTT. Only ALGO_SHOUP
// is instantiated (the only algo the rescale path uses). dat = limbptr + (limbsize-2), so
// dat[0] = the qb limb, dat[1] = the qa top; primeid_rescale = qa's primeid, and qb's primeid
// is derived as primeid_rescale - 1 (single-GPU composite chains have level-ordered q ids —
// asserted by the host caller, which falls back to two passes otherwise).
// NTT_RESCALEK (RR, TO-TRY §2.10a): the same idea generalized to a k-way drop, for the
// RATIONAL RESCALING chain — where one payload rescale divides out 3-4 primes and the drops
// sit at a window EDGE, not at the top of a prefix. Two things change versus RESCALE2:
//   - the dropped primeids are arbitrary and there are up to RESCALEK_MAX of them, so they
//     ride in `primeid_rescale` PACKED (see rescaleK_pack) instead of being derived as
//     {r, r-1}, and
//   - the k dropped limbs' coeff-domain data is addressed through `dat` itself: the host
//     passes a device array of exactly the k dropped-limb pointers IN DROP ORDER, and stage 1
//     receives that array (not one limb) so it can read all k with the same transposed,
//     coalesced pattern.
// The arithmetic is the k-term generalization of rescale2_combine and is likewise
// bit-identical to k sequential NTT_RESCALE passes — see rescaleK_combine (NTT.cu) for the
// recursion and why sequential rounding is preserved. ALGO_SHOUP only, like RESCALE2.
enum NTT_MODE {
    NTT_NONE,
    NTT_RESCALE,
    NTT_MULTPT,
    NTT_MODDOWN,
    NTT_KSK_DOT,
    NTT_KSK_DOT_ACC,
    NTT_RESCALE2,
    NTT_RESCALEK
};

/** How many primes one NTT_RESCALEK pass can divide out. Bounded by the packing below (3 bits
 *  of count + RESCALEK_MAX * 6 bits of primeid must fit an int) and by register pressure in
 *  the stage-1 combine, which keeps k live residues per coefficient lane. A rescale that drops
 *  more is split into consecutive groups of at most this many by the host. */
constexpr int RESCALEK_MAX = 4;
static_assert(MAXP <= 64, "rescaleK packs each primeid into 6 bits");

/** The dropped set travels in the single `primeid_rescale` kernel argument: count in bits
 *  [0,3), then one 6-bit primeid per drop. Order IS the drop order and is not free to permute
 *  — see rescaleK_combine. */
__host__ __device__ __forceinline__ int rescaleK_count(const int packed) {
    return packed & 7;
}
__host__ __device__ __forceinline__ int rescaleK_id(const int packed, const int t) {
    return (packed >> (3 + 6 * t)) & (MAXP - 1);
}
inline int rescaleK_pack(const int* d, const int k) {
    int packed = k;
    for (int t = 0; t < k; ++t)
        packed |= (d[t] & (MAXP - 1)) << (3 + 6 * t);
    return packed;
}

template <typename T, bool second = true, ALGO algo = ALGO_SHOUP, NTT_MODE mode = NTT_NONE>
__global__ void NTT_(const Global::Globals* Globals, T* __restrict__ dat, const int __grid_constant__ primeid,
                     T* __restrict__ res, const T* __restrict__ pt = nullptr,
                     const int __grid_constant__ primeid_rescale = -1, T* __restrict__ res2 = nullptr,
                     const T* __restrict__ kskb = nullptr);

template <bool second, ALGO algo, NTT_MODE mode>
__global__ void NTT_(const Global::Globals* Globals, void** __restrict__ dat, const int __grid_constant__ primeid_init,
                     void** __restrict__ res, void** __restrict__ pt = nullptr,
                     const int __grid_constant__ primeid_rescale = -1, void** __restrict__ res2 = nullptr,
                     void** __restrict__ kskb = nullptr);

// E1 (Phase 3b): fused two-stage NTT/INTT via one cooperative launch (plain
// NTT_NONE/INTT_NONE, ALGO_SHOUP only — see NTT.cu). Returns true iff the fused pair was
// launched (possibly gy-chunked to respect cooperative residency); false = caller must run
// the classic two-pass launches. Preconditions checked inside: FIDESLIB_FUSED_NTT != 0
// (default on), cooperative-launch support, stream not capturing, capacity ≥ gridDim.x.
// The CALLER must only offer stage-shape-equal transforms (even logN: blockDim and
// gridDim.x identical across stages) — pass that single grid.x/block/bytes.
bool launchFusedNTTPair(bool inverse, const Global::Globals* Globals, void** dat, int primeid_init, int num_limbs,
                        dim3 grid_x_only, dim3 block, int bytes, void** aux, void** out, cudaStream_t stream);

// ------------------------------------- 1D NTT version ----------------------------------------

template <typename T, int WARP_SIZE = 32>
__global__ void NTT_1D(const Global::Globals* Globals, T* dat, const T* psi_dat, const int __grid_constant__ N,
                       const int __grid_constant__ primeid, const int __grid_constant__ logN);

template <typename T, int WARP_SIZE = 32>
__global__ void INTT_1D(const Global::Globals* Globals, T* dat, const T* psi_dat, const int __grid_constant__ N,
                        const int __grid_constant__ primeid, const T N_inv, const int __grid_constant__ logN);
}  // namespace FIDESlib

#endif  //FIDESLIB_NTT_CUH
