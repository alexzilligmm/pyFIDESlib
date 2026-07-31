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
// ── SELECTING A VARIANT: one knob, three values ───────────────────────────────────────────
//   FIDESLIB_NTT_SWIZZLE = 0   OFF        41.18 % conflicts   baseline
//                          1   GRAY        9.09 %             DEFAULT — banked, -0.19 ms/bts
//                          2   GRAY+ROW    0.00 % (A-rows)    measured WORSE than 1 (+0.185 ms)
// It is ONE knob on purpose: the variants differ in both swz_base and the int4 lane permute, and
// independent macros made illegal combinations reachable (an OFF build that still applied Gray's
// lane reorder would silently swap two lanes of every int4 store).
//
// Measured on n32/sm_120, 25-pair A/B (`scripts/bin_ab.sh`, read the median-of-iters column):
//   1 vs 0 : mean -0.192 +/- 0.052 ms (t=-3.67, p=0.0002, 18/25) — BANKED, default
//   2 vs 1 : mean +0.185 +/- 0.082 ms (1/4)  — the row term removes the remaining A-row conflicts
//            (NTT store 10.3 -> 1.3 %, INTT load 20.1 -> 14.5 %) but costs +12.4 % instructions,
//            and this kernel pays ~1:1 for instructions. Kept only as an escape hatch.
// Exchange rate for pricing any future variant BEFORE building it: +1 % instructions costs
// ~0.016 ms, -1 point of conflicts gains ~0.007 ms => need > 2.3 conflict-points per 1 %.
#ifndef FIDESLIB_NTT_SWIZZLE
#define FIDESLIB_NTT_SWIZZLE 1
#endif
#if FIDESLIB_NTT_SWIZZLE < 0 || FIDESLIB_NTT_SWIZZLE > 2
#error "FIDESLIB_NTT_SWIZZLE must be 0 (off), 1 (Gray, default) or 2 (Gray + row term)"
#endif

// SPLIT FORM — this is what makes the swizzle affordable. pos(i,e) factors exactly into a
// row-invariant part and a per-row part, because the quad term lives in bits >=2 and the lane
// terms in bits 0..1, so the carries never interact and the sum becomes an XOR:
//     pos(i,e) = swz_base(e) ^ swz_row(i)
// In every hot loop `e` is loop-invariant across the unrolled i-loop, so swz_base is computed
// ONCE per stage and swz_row(i) folds into an immediate. A FUSED form cost +23.7 % instructions
// and ate the entire conflict win; do not re-fuse it.
//
// swz_base = the Gray code, e ^ (e>>1). Two ops. It is the optimum of THREE independent searches:
// the cute::Swizzle<B,M,S> family (it IS Swizzle<5,0,1>), a 2104-map chess-style magic-multiplier
// search, and an exhaustive Pareto sweep of unit lower-triangular GF(2) maps (9.09 % at cost 2,
// nothing at cost 3-9 beats it). A hand-derived 5-op map scores IDENTICALLY (9.09 %) and was
// removed as strictly dominated — see the ledger for its derivation, which is what explains the
// constraint: each butterfly stage reads one hyperplane {g_s = v} of the quad cube, so a map that
// fixes ALL stages needs kernel exactly {0000,1111}.
template <typename T>
__device__ __forceinline__ int swz_base(const int e) {
    if constexpr (FIDESLIB_NTT_SWIZZLE >= 1 && sizeof(T) == 4) {
        return e ^ (e >> 1);
    } else {
        return e;
    }
}

// The ROW term is the only part with a PER-ACCESS cost (one LOP3 on every shared access). It buys
// the last 9.09 % -> 0 % — the transposed accesses, which differ only in the ROW and so cannot be
// separated by anything row-invariant — and measured a net LOSS. Off unless FIDESLIB_NTT_SWIZZLE=2.
template <typename T>
__device__ __forceinline__ int swz_row(const int i) {
    if constexpr (FIDESLIB_NTT_SWIZZLE == 2 && sizeof(T) == 4) {
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

// The lane permutation code for that quad. NOTE this is NOT a plain XOR under Gray — it is
// gray2(r) ^ c with gray2 = [0,1,3,2]; swz_perm4 below consumes `c` and applies both halves.
// e is masked to its quad base first: c is the permutation of the QUAD, not the physical lane of
// e itself, and those coincide only when e is already quad-aligned. Returns 0 on the u64 path.
template <typename T>
__device__ __forceinline__ int swz_lx(const int i, const int e) {
    return swz_pos<T>(i, e & ~3) & 3;
}

// Reorder the four register lanes of an int4 so logical lane r lands at physical lane
// gray2(r) ^ c. Inverting: out.lane[l] = v.lane[gray2(l ^ c)].
__device__ __forceinline__ int4 swz_perm4(int4 v, [[maybe_unused]] const int c) {
#if FIDESLIB_NTT_SWIZZLE == 0
    return v;                                   // must be the exact identity, not a lane shuffle
#elif FIDESLIB_NTT_SWIZZLE == 1
    // Without the row term c is only 0 or 2, so this is a FIXED 2<->3 swap (free — a static
    // reorder) composed with a conditional half-swap: 4 SEL.
    return c ? make_int4(v.w, v.z, v.x, v.y) : make_int4(v.x, v.y, v.w, v.z);
#else
    // With the row term c spans 0..3: a conditional base perm, then a conditional half-swap. 8 SEL.
    const int4 b = (c & 1) ? make_int4(v.y, v.x, v.z, v.w) : make_int4(v.x, v.y, v.w, v.z);
    return (c & 2) ? make_int4(b.z, b.w, b.x, b.y) : b;
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
