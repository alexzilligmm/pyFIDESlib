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
//                          2   GRAY+ROW    0.00 % (A-rows)    measured WORSE than 1 (+0.185 ms;
//                                                              RE-TRIED on top of WARP_SHFL and
//                                                              still worse, +0.146 ms — FAILURE §2.6)
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

// ── TO-TRY §2.2 candidate 1: WARP-SHUFFLE BUTTERFLIES for the intra-warp stages ───────────
// The last five forward stages (m = 16,8,4,2,1) and the first five inverse ones are entirely
// INTRA-WARP: at m <= 16 a warp's 32 threads cover exactly one aligned 64-element block of a
// row (thread tid owns {ins0(tid,s), ins0(tid,s)+2^s}, and ins0 only permutes the low 6 bits).
// So those stages need no shared memory at all — the pair lives in two registers and the
// per-stage RE-ASSIGNMENT is one `__shfl_xor`.
//
// Why one shuffle is enough. Write an element as (H, b_s, b_{s-1}, Lo). At stage s the four
// elements with a given (H,Lo) sit in the two lanes that differ in lane bit s-1 (the lane index
// is the element index with bit s deleted, so b_{s-1} lands at position s-1); at stage s-1 the
// lane index is the element index with bit s-1 deleted, so b_s lands at position s-1 — the SAME
// two lanes. Lane with bit(s-1)=0 keeps its low element and trades its high one; the other keeps
// its high element and trades its low one. The map is an involution, so ONE helper serves both
// the descending (CT) and the ascending (GS) loop. Verified exhaustively against the
// shared-memory ownership formula for all 32 lanes and all five stages, both directions.
//
// Cost model (u32, M = 8 rows, blockDim 128): the five stages were 5*M*(2 LDS + 2 STS) = 160
// shared accesses per thread plus 5 __syncwarp; they become 1 entry pair-load + 1 exit
// pair-store per row (2*M*(1 LDS + 1 STS) = 32, i.e. exactly what the two boundary stages
// already paid) plus 4*M shuffles. Net: ~128 shared accesses per thread traded for 32 SHFL,
// on a kernel measured LATENCY-bound (DRAM 1.5 %, 12.4 warp-cycles per issued instruction) —
// i.e. it is spending the resource that is actually scarce. It also removes the residual
// butterfly bank conflicts (the Gray swizzle's remaining 9.09 %) for those stages, since they
// no longer touch shared memory.
//
// Bit-exact by construction: same values, same twiddles, same order of operations — only the
// medium of exchange changes. Gated anyway: [bitcmp] add/mult/mult2-rescaled every badcount 0,
// test_composite_bootstrap core 2/2, no register spills (STACK:0; the shipping ALGO_SHOUP pair
// goes 40 -> 40 and 42 -> 46 regs).
//   FIDESLIB_NTT_WARP_SHFL = 0  shared-memory butterflies at every stage (exact revert)
//                            1  registers + __shfl_xor for the five intra-warp stages — DEFAULT
//
// MEASURED (2026-07-31, n32/sm_120, `scripts/bin_ab.sh` 12 alternating pairs x ITERS=100):
//   median-of-iters  -0.352 ms  SE 0.085  t=-4.14  11/12 pairs   (-0.31 ms, t=-3.85, after the
//   min              -0.457 ms  SE 0.030  t=-15.4  12/12 pairs    drop below)
//   mean             -0.131 ms  <- CONTAMINATED, see below
// One pair dropped by a stated rule: within-run sd > 3x the median within-run sd (0.49 ms). Pair
// 6's B run had within-run sd 15.08 ms against 0.35-0.79 everywhere else; that ONE excursion is
// the entire mean/median gap, i.e. the mean estimator would have called this lever noise.
// Realized sd(diff) on the median column = 0.294 ms over 12 pairs.
// Requires blockDim.x >= 32 (a full warp per exchange); guarded at runtime, and blockDim.x is
// 1 << ((logN+1)/2 - 1) = 128 at logN 16.
#ifndef FIDESLIB_NTT_WARP_SHFL
#define FIDESLIB_NTT_WARP_SHFL 1
#endif
#if FIDESLIB_NTT_WARP_SHFL < 0 || FIDESLIB_NTT_WARP_SHFL > 1
#error "FIDESLIB_NTT_WARP_SHFL must be 0 (off) or 1 (on)"
#endif

// Number of trailing stages run in registers, and the block size below which the path is
// disabled (one exchange spans lane bits 0..3, so the warp must be full).
#define NTT_SHFL_STAGES 5

// ── DIAGNOSTIC ONLY — NEVER SHIP THIS AT 1. TO-TRY §2.2 candidates 2 (EOT twiddles) and 3
// (twiddle coalescing) both target ONE site: the middle-scale twiddle of the 4-step transform
// (`NTT__`'s !second epilogue and `INTT__`'s second prologue), which costs 2*M = 16 scattered
// GLOBAL gathers plus 16 scattered SHARED reads per thread — the latter being the residual
// ~15 % NTT load conflict the ledger already localised.
//
// Setting this to 1 replaces that twiddle with a block-uniform register constant: every gather,
// every shared read and all the exponent arithmetic vanish, while every modmult that consumes
// the twiddle stays. **Results are WRONG by construction** — this is the ledger's "ablate the
// stream to bound the prize BEFORE building the compression" method, and the resulting wall
// delta is the UPPER BOUND on candidates 2 and 3 combined. If that bound is small, both close
// without anyone writing an EOT.
#ifndef FIDESLIB_NTT_TWIDDLE_ABLATE
#define FIDESLIB_NTT_TWIDDLE_ABLATE 0
#endif

// ── DIAGNOSTIC ONLY — NEVER SHIP THIS AT 1. Bounds the LAST open item in TO-TRY §2.2: TMA for
// the strided tile moves. The stage-1 transposed LOAD (`NTT__`) and the final transposed STORE
// (`INTT__`) are the only global<->shared accesses with real per-thread address math
// (IMAD/LEA/SHF over col_init, gridDim.x, blockIdx.x, j&2) — and the biggest launch is
// ISSUE-limited, not bandwidth-limited, which is the whole argument for moving that math into
// a copy engine.
//
// Setting this to 1 replaces those two indices with a plain linear one — same access COUNT,
// same int4 width, same bytes moved, but no address arithmetic and a fully coalesced pattern.
// **Results are WRONG by construction** (the transpose is what makes the 4-step NTT correct).
// It removes strictly MORE than TMA could recover, so the wall delta is a hard upper bound on
// the whole TMA rewrite — which is worth a day of descriptor plumbing only if this is large.
// The shared-side indexing (AS / swz_*) is deliberately left untouched, so this isolates the
// GLOBAL side alone.
#ifndef FIDESLIB_NTT_TRANSPOSE_ABLATE
#define FIDESLIB_NTT_TRANSPOSE_ABLATE 0
#endif

// ── TO-TRY §2.2 candidate 2: EXTENDED ON-THE-FLY TWIDDLES (EOT) at the middle-scale site ──
// The ablation above bounded this site at −0.910 ms (12/12 pairs, t=−14.3), so it is worth
// generating rather than loading.
//
// The site multiplies element (i, k) by W(exp) with exp = block_pos·br_j, block_pos =
// blockIdx.x·M + i, and W(e) = psi_no[2e] = ω^e. **exp is AFFINE in i**, so
//     W(exp_i) = W(blockIdx.x·M·br_j) · W(br_j)^i
// i.e. one base value per k plus a running multiply down the M loop replaces M scattered
// lookups. W is a group homomorphism into Z_q, so `W(a)·W(b) = W(a+b)` holds EXACTLY in modular
// arithmetic — the transform stays bit-identical, which is what makes [bitcmp] the right gate.
//
// Per thread this turns 2·M scattered global gathers + 2·M scattered shared reads into 4 global
// loads + 2 shared reads + 2·M Barrett multiplies. The advance uses ALGO_BARRETT because there
// is no `psi_no_shoup` table to feed a Shoup multiply — that is the EOT bargain (a table load
// traded for ALU), and this kernel is latency-bound with DRAM at 1.5 %.
//
// It also removes the `psi[hi_exp_br]` scattered SHARED read from the inner loop (2 reads
// instead of 2·M), which is the residual ~15 % NTT *load* conflict the ledger localised —
// so candidate 3 (twiddle coalescing) is collected by the same change rather than separately.
//
// FIDESlib already applies exactly this trick in `forward/backward_negacyclic_scale` (one load,
// then `aux *= root` across M). This extends it to the one site that was still table-fed.
//   FIDESLIB_NTT_EOT = 0  per-element table lookups (exact revert)
//                      1  generate by iterative multiplication
#ifndef FIDESLIB_NTT_EOT
#define FIDESLIB_NTT_EOT 1
#endif
#if FIDESLIB_NTT_EOT < 0 || FIDESLIB_NTT_EOT > 1
#error "FIDESLIB_NTT_EOT must be 0 (off) or 1 (on)"
#endif

template <typename T>
__device__ __forceinline__ T shfl_xor_(const T v, const int lane_mask) {
    if constexpr (sizeof(T) == 8) {
        return (T)__shfl_xor_sync(0xFFFFFFFFu, (unsigned long long)v, lane_mask);
    } else {
        return (T)__shfl_xor_sync(0xFFFFFFFFu, (unsigned int)v, lane_mask);
    }
}

// Move the register-held butterfly pair from the stage whose paired bit is s to the one whose
// paired bit is s-1 (or back — the map is its own inverse). `s_lo` = min(s, s-1) = the LANE bit
// that separates the two partners = log2 of the stage being entered (forward) / left (inverse).
template <typename T>
__device__ __forceinline__ void warp_pair_exchange(T& a0, T& a1, const int tid, const int s_lo) {
    const int t = (tid >> s_lo) & 1;
    const T got = shfl_xor_<T>(t ? a0 : a1, 1 << s_lo);
    if (t)
        a0 = got;
    else
        a1 = got;
}

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
#if FIDESLIB_NTT_SMR
            if constexpr (sizeof(T) == 4)
                b = modmult<ALGO_SMR>(b, psi, primeid, shoup_psi);  // shoup slot carries the MONT twiddle
            else
                b = modmult<algo>(b, psi, primeid, shoup_psi);
#else
            b = modmult<algo>(b, psi, primeid, shoup_psi);
#endif
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
#if FIDESLIB_NTT_SMR
            if constexpr (sizeof(T) == 4)
                d = modmult<ALGO_SMR>(b, psi, primeid, shoup_psi);  // shoup slot carries the MONT twiddle
            else
                d = modmult<algo>(b, psi, primeid, shoup_psi);
#else
            d = modmult<algo>(b, psi, primeid, shoup_psi);
#endif
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
