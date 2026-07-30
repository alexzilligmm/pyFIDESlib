//
// Created by carlosad on 21/10/24.
//

#ifndef GPUCKKS_NTTHELPER_CUH
#define GPUCKKS_NTTHELPER_CUH

namespace FIDESlib {
    template<typename T>
    __device__ __inline__ void swap(T &a, T &b) {
        T c = a;
        a = b;
        b = c;
    }

#define A(i) \
 (((T *) buffer) + 2 * blockDim.x * (i) )

// ── n32 speed: shared-memory PARITY SWIZZLE for the NTT/INTT rows (u32 only) ──────────────
// Two distinct degeneracies in the ADDRESS SET, neither fixable by padding (both are
// INTRA-array, so widening the row stride shifts every address by the same amount):
//   (1) the butterfly index j1 = "insert a 0 bit at position s into tid" — at every stage
//       m = 2^s < 32 the low 5 address bits carry only 4 bits of tid, so a warp touches
//       exactly 16 banks: an inherent 2-way conflict at m = 1,2,4,8,16.
//   (2) the transposed stage-1 load / stage-N store, where threads differ mainly in the ROW
//       and rows sit 2*blockDim.x elements apart == 0 (mod 32 banks): 4-way, measured 75%.
//
// Logical element e = 4g+r of row i physically lives at
//     quad = (g ^ 7*bit3(g)) ^ (i & 7)      <- (1): kernel of the g-map is exactly {0,1111},
//                                              so it stays injective on EVERY hyperplane
//                                              {g_k = v} — and each butterfly stage reads
//                                              precisely one such hyperplane.
//     lane = r ^ 3*bit0(g) ^ bit2(i)        <- (2): separates the quad-pairs that (1) folds
//                                              together, and the two rows the transposed
//                                              accesses pair up.
//     pos  = 4*quad + lane
// The quad map is a bijection of each aligned 64-element block, so it never leaves the row,
// and it keeps every 4-element group contiguous and 16-B aligned: the int4/int2 accesses
// SURVIVE — only their register lanes get permuted (swz_perm4 / the pos^1 pair swap).
// Verified exhaustively against a bank model (every stage, both directions, both blockDim
// 128 and 256): every u32 shared access in NTT__/INTT__ drops to ZERO conflicts, from 41%.
//
// u64 rows are deliberately LEFT ALONE (identity): an 8-byte element makes bank = (2e)%32,
// which reaches only 16 banks whatever the permutation, and this swizzle modelled WORSE
// there (45.8% -> 48.0%). Do not "unify" the two paths.
//
// Build with -DFIDESLIB_NTT_SWIZZLE=0 to get an EXACT revert (all three helpers collapse to
// the identity, the pos^1 pair swaps become no-ops because j is even, and swz_perm4 is fed 0).
// That is what the wall A/B alternates, and it is the escape hatch if a future arch regresses.
#ifndef FIDESLIB_NTT_SWIZZLE
#define FIDESLIB_NTT_SWIZZLE 1
#endif

// SPLIT FORM — this is what makes the swizzle affordable. pos(i,e) factors exactly into a
// row-invariant part and a per-row part, because the quad term lives in bits >=2 and the two
// lane terms in bits 0..1, so the carries never interact and the sum becomes an XOR:
//     pos(i,e) = swz_base(e) ^ swz_row(i)
// In every hot loop `e` is loop-invariant across the unrolled i-loop, so swz_base is computed
// ONCE and swz_row(i) folds into an immediate — the whole per-access cost is one LOP3.
// The fused form cost +23.7% instructions and ate the entire conflict win; do not re-fuse it.
// GRAY variant: swz_base(e) = e ^ (e>>1). Two ops instead of five, at 15.4% residual conflicts
// instead of 14.3% — i.e. slightly WORSE on the counter and much cheaper on the ALU, which is the
// trade that matters here (this kernel pays ~1:1 for instructions). It is the optimum of both the
// cute::Swizzle<B,M,S> family (= Swizzle<5,0,1>) and of a 2104-map magic-multiplier search, which
// both independently land on it. Mutually exclusive with the ROW term.
#ifndef FIDESLIB_NTT_SWIZZLE_GRAY
#define FIDESLIB_NTT_SWIZZLE_GRAY 1
#endif
#if FIDESLIB_NTT_SWIZZLE_GRAY && FIDESLIB_NTT_SWIZZLE_ROW
#error "FIDESLIB_NTT_SWIZZLE_GRAY and _ROW are mutually exclusive (the Gray lane map is not an XOR)"
#endif

template <typename T>
__device__ __forceinline__ int swz_base(const int e) {
    if constexpr (FIDESLIB_NTT_SWIZZLE && sizeof(T) == 4) {
#if FIDESLIB_NTT_SWIZZLE_GRAY
        return e ^ (e >> 1);
#else
        const int g = e >> 2;
        return 4 * (g ^ (-((g >> 3) & 1) & 7)) + ((e & 3) ^ (-(g & 1) & 3));
#endif
    } else {
        return e;
    }
}

// The ROW term is separately switchable and DEFAULTS OFF, because it is the only part of the
// swizzle with a per-access cost. swz_base is loop-invariant across the unrolled i-loop (the
// butterfly computes it once per stage and reuses it for all M rows), so quad-only is FREE;
// the row term adds one LOP3 to every shared access, and at this kernel's arithmetic intensity
// that measured +16.5% instructions — more than the conflicts it removes are worth.
// It buys the last 14.3% -> 0% (the transposed accesses, which differ only in the ROW and so
// cannot be separated by anything row-invariant). Enable only if smem ever becomes the limiter.
#ifndef FIDESLIB_NTT_SWIZZLE_ROW
#define FIDESLIB_NTT_SWIZZLE_ROW 0
#endif

template <typename T>
__device__ __forceinline__ int swz_row(const int i) {
    if constexpr (FIDESLIB_NTT_SWIZZLE && FIDESLIB_NTT_SWIZZLE_ROW && sizeof(T) == 4) {
        return (4 * (i & 7)) ^ ((i >> 2) & 1);
    } else {
        return 0;
    }
}

template <typename T>
__device__ __forceinline__ int swz_pos(const int i, const int e) {
    return swz_base<T>(e) ^ swz_row<T>(i);
}

// int4 slot (quad index) holding the logical quad that contains element e of row i.
template <typename T>
__device__ __forceinline__ int swz_quad(const int i, const int e) {
    return swz_pos<T>(i, e) >> 2;
}

// Lane permutation inside that quad: logical lane r is stored at physical lane r ^ swz_lx.
template <typename T>
__device__ __forceinline__ int swz_lx(const int i, const int e) {
    // Mask e down to its quad base first: this must be the lane XOR (the permutation), not the
    // physical lane of e itself — those coincide only when e is already quad-aligned. Also
    // keeps the u64 path returning 0, as swz_perm4 requires.
    return swz_pos<T>(i, e & ~3) & 3;
}

// out.lane[l] = v.lane[l ^ x], i.e. logical lane r lands at physical lane r ^ x. Involutive,
// so the same call reorders both a store (logical -> physical) and a load (physical -> logical).
__device__ __forceinline__ int4 swz_perm4(int4 v, const int x) {
#if FIDESLIB_NTT_SWIZZLE_GRAY
    // Gray lane map is gray2(r) ^ c with gray2 = [0,1,3,2] and c = x = 2*(q&1). Inverting it:
    // out.lane[l] = v.lane[gray2(l ^ c)], which is a FIXED 2<->3 swap (free, a static reorder)
    // composed with a conditional half-swap. Still 4 SEL, same as the XOR case.
    return x ? make_int4(v.w, v.z, v.x, v.y) : make_int4(v.x, v.y, v.w, v.z);
#elif !FIDESLIB_NTT_SWIZZLE_ROW
    // With the row term off the lane XOR is only ever 0 or 3, i.e. a plain 4-way reversal.
    // Telling the compiler that halves this to 4 SEL instead of 8 — and the SELs on the
    // vector paths are where quad-only's whole instruction overhead lives (SASS: +32 SEL).
    return x ? make_int4(v.w, v.z, v.y, v.x) : v;
#else
    if (x & 1) {
        int t = v.x; v.x = v.y; v.y = t;
        t = v.z; v.z = v.w; v.w = t;
    }
    if (x & 2) {
        int t = v.x; v.x = v.z; v.z = t;
        t = v.y; v.y = v.w; v.w = t;
    }
    return v;
#endif
}

// Swizzled scalar access to logical element `e` of shared row `i`.
#define AS(i, e) (A(i)[FIDESlib::swz_base<T>(e) ^ FIDESlib::swz_row<T>(i)])

#define OFFSET_T(i) \
 ((blockDim.x * 2 * M) * blockIdx.x + 2 * blockDim.x * (i) + 2 * threadIdx.x)

#define OFFSET_2T(i) \
 ((blockDim.x * M) * blockIdx.x +  blockDim.x * (i) + threadIdx.x)

    template<typename T, ALGO algo = ALGO_SHOUP>
    __device__ __forceinline__ void CT_butterfly(T &c, T &d, T psi, const int primeid, T shoup_psi = 2) {
        T a = c;
        T b = d;
        if constexpr (algo == 1) {

        } else if constexpr (algo == 2) {
            const uint64_t hi = __umul64hi(b, shoup_psi);
            b = b * psi - hi * C_.primes[primeid];
            d = a - b;
            c = a + b;
        } else if constexpr (algo == 3) {
            b = modmult<algo>(b, psi, primeid, shoup_psi);
            c = modadd(a, b, primeid);
            d = modsub(a, b, primeid);
        } else if constexpr (algo <= 5) {
            // assert(b < primes[primeid]);
            // assert(a < primes[primeid]);
            //  T baux = modmult<0>(b, psi, primeid);
            b = modmult<algo>(b, psi, primeid);
            // assert(psi < primes[primeid]);
            // assert(b < primes[primeid]);
            // assert(b == baux);
            c = modadd(a, b, primeid);
            d = modsub(a, b, primeid);
        }
    }

    template<typename T, ALGO algo = ALGO_SHOUP>
    __device__ __forceinline__ void GS_butterfly(T &c, T &d, T psi, const int primeid, T shoup_psi = 2) {
        T a = c;
        T b = d;
        if constexpr (algo == 1) {
        } else if constexpr (algo == 2) {
            d = a - b;
            c = a + b;
            const uint64_t hi = __umul64hi(d, shoup_psi);
            d = d * psi - hi * C_.primes[primeid];
        } else if constexpr (algo == 3) {
            c = modadd(a, b, primeid);
            b = modsub(a, b, primeid);
            d = modmult<algo>(b, psi, primeid, shoup_psi);
        } else if constexpr (algo <= 5) {
            //      assert(b < primes[primeid]);
            //      assert(a < primes[primeid]);
            c = modadd(a, b, primeid);
            b = modsub(a, b, primeid);
            //   T baux = modmult<0>(b, psi, primeid);
            d = modmult<algo>(b, psi, primeid);
            //     assert(psi < primes[primeid]);
            //     assert(d < primes[primeid]);
            //     assert(d == baux);
        }
    }

}
#endif //GPUCKKS_NTTHELPER_CUH
