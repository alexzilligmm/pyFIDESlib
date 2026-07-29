//
// Created by carlosad on 27/09/24.
//

#include <printf.h>
#include <stdexcept>
#include <string>
#include "CKKS/ElemenwiseBatchKernels.cuh"
#include "CKKS/KskSeedExpand.cuh"
#include "CKKS/Rescale.cuh"
#include "Rotation.cuh"

#include <cooperative_groups.h>
#include <cuda/barrier>
//#include "cooperative_groups/memcpy_async.h"
namespace cg = cooperative_groups;

namespace FIDESlib {
namespace CKKS {
__global__ void mult1AddMult23Add4_(const __grid_constant__ int primeid_init, void** l, void** l1, void** l2, void** l3,
                                    void** l4) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr ALGO algo = ALGO_BARRETT;

    if (ISU64(primeid)) {
        using T = uint64_t;
        T aux = ((T*)l4[blockIdx.y])[idx];
        T res = modmult<algo>(((T*)l[blockIdx.y])[idx], ((T*)l1[blockIdx.y])[idx], primeid);
        res = modadd(res, aux, primeid);
        res = modadd(res, modmult<algo>(((T*)l2[blockIdx.y])[idx], ((T*)l3[blockIdx.y])[idx], primeid), primeid);
        ((T*)l[blockIdx.y])[idx] = res;
    } else {
        using T = uint32_t;
    }
}

__global__ void multnomoddownend_(const __grid_constant__ int primeid_init, void** c1, void** c0, void** bc0,
                                  void** bc1, void** in, void** aux) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr ALGO algo = ALGO_BARRETT;

    if (ISU64(primeid)) {
        using T = uint64_t;
        T aux0 = ((T*)in[blockIdx.y])[idx];
        T res = modmult<ALGO_SHOUP>(modmult<algo>(((T*)c1[blockIdx.y])[idx], ((T*)bc0[blockIdx.y])[idx], primeid),
                                    C_.P[primeid], primeid, C_.P_shoup[primeid]);
        res = modadd(res, aux0, primeid);
        res = modadd(res,
                     modmult<ALGO_SHOUP>(modmult<algo>(((T*)c0[blockIdx.y])[idx], ((T*)bc1[blockIdx.y])[idx], primeid),
                                         C_.P[primeid], primeid, C_.P_shoup[primeid]),
                     primeid);
        ((T*)c1[blockIdx.y])[idx] = res;
        aux0 = ((T*)aux[blockIdx.y])[idx];
        res = modmult<ALGO_SHOUP>(modmult<algo>(((T*)c0[blockIdx.y])[idx], ((T*)bc0[blockIdx.y])[idx], primeid),
                                  C_.P[primeid], primeid, C_.P_shoup[primeid]);
        res = modadd(res, aux0, primeid);

        ((T*)c0[blockIdx.y])[idx] = res;
    } else {
        using T = uint32_t;
    }
}

__global__ void mult1Add2_(const __grid_constant__ int primeid_init, void** l, void** l1, void** l2) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr ALGO algo = ALGO_BARRETT;

    if (ISU64(primeid)) {
        using T = uint64_t;
        T aux = ((T*)l2[blockIdx.y])[idx];
        T res = modmult<algo>(((T*)l[blockIdx.y])[idx], ((T*)l1[blockIdx.y])[idx], primeid);
        ((T*)l[blockIdx.y])[idx] = modadd(res, aux, primeid);
    } else {
        using T = uint32_t;
        T aux = ((T*)l2[blockIdx.y])[idx];
        T res = modmult<algo>(((T*)l[blockIdx.y])[idx], ((T*)l1[blockIdx.y])[idx], primeid);
        ((T*)l[blockIdx.y])[idx] = modadd(res, aux, primeid);
    }
}

template <typename T>
__device__ __forceinline__ void addMult__(T* l, const T* l1, const T* l2, const int primeid) {
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr ALGO algo = ALGO_BARRETT;

    l[idx] = modadd(l[idx], modmult<algo>(l1[idx], l2[idx], primeid), primeid);
}

template <typename T>
__global__ void addMult_(T* l, const T* l1, const T* l2, const __grid_constant__ int primeid) {
    addMult__<T>(l, l1, l2, primeid);
}

__global__ void addMult_(void** l, void** l1, void** l2, const __grid_constant__ int primeid_init) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];

    //    if (threadIdx.x + blockDim.x * blockIdx.x == 0)
    //        printf("%d %d\n", primeid_init + blockIdx.y, primeid);
    if (ISU64(primeid)) {
        addMult__<uint64_t>((uint64_t*)l[blockIdx.y], (uint64_t*)l1[blockIdx.y], (uint64_t*)l2[blockIdx.y], primeid);
    } else {
        addMult__<uint32_t>((uint32_t*)l[blockIdx.y], (uint32_t*)l1[blockIdx.y], (uint32_t*)l2[blockIdx.y], primeid);
    }
}

__global__ void Mult_(void** l, void** l1, void** l2, const __grid_constant__ int primeid_init) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    //    if (idx == 0)
    //        printf("%d %d\n", primeid_init + blockIdx.y, primeid);
    if (ISU64(primeid)) {
        ((uint64_t*)l[blockIdx.y])[idx] =
            modmult<ALGO_BARRETT>(((uint64_t*)l1[blockIdx.y])[idx], ((uint64_t*)l2[blockIdx.y])[idx], primeid);
    } else {
        ((uint32_t*)l[blockIdx.y])[idx] =
            modmult<ALGO_BARRETT>(((uint32_t*)l1[blockIdx.y])[idx], ((uint32_t*)l2[blockIdx.y])[idx], primeid);
    }
}

__global__ void square_(void** l, void** l1, const __grid_constant__ int primeid_init) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (ISU64(primeid)) {
        uint64_t in = ((uint64_t*)l1[blockIdx.y])[idx];
        ((uint64_t*)l[blockIdx.y])[idx] = modmult<ALGO_BARRETT>(in, in, primeid);
    } else {
        uint32_t in = ((uint32_t*)l1[blockIdx.y])[idx];
        ((uint32_t*)l[blockIdx.y])[idx] = modmult<ALGO_BARRETT>(in, in, primeid);
    }
};

__global__ void binomial_square_fold_(void** c0_res, void** c2_key_switched_0, void** c1, void** c2_key_switched_1,
                                      const __grid_constant__ int primeid_init) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];

    if (ISU64(primeid)) {
        uint64_t in2_0 = ((uint64_t*)c2_key_switched_0[blockIdx.y])[idx];
        uint64_t in2_1 = ((uint64_t*)c2_key_switched_1[blockIdx.y])[idx];
        uint64_t in0 = ((uint64_t*)c0_res[blockIdx.y])[idx];
        uint64_t ok = modadd(modmult<ALGO_BARRETT>(in0, in0, primeid), in2_0, primeid);
        ((uint64_t*)c0_res[blockIdx.y])[idx] = ok;
        uint64_t in1 = ((uint64_t*)c1[blockIdx.y])[idx];
        uint64_t aux = modmult<ALGO_BARRETT>(in0, in1, primeid);
        uint64_t aux2 = modadd(aux, aux, primeid);
        ok = modadd(aux2, in2_1, primeid);
        ((uint64_t*)c1[blockIdx.y])[idx] = ok;
    } else {
    }
}

// n32: this is the bootstrap's MODULUS RAISE. The old body was guarded by
// `ISU64(primeid) && ISU64(0)` with NO else, so on a uniform-U32 chain (constants.type == 0,
// i.e. ISU64 false for every prime) it wrote NOTHING — and the target limbs were freshly
// allocated by RNSPoly::grow -> generate() WITHOUT being zeroed, so EvalMod consumed
// uninitialized pool memory and the bootstrap decrypted to NaN.
// Handle every width combination: read at the source limb's width, switch modulus in a type
// wide enough to hold both moduli, store at the target limb's width. The mixed cases are not
// exercised by a uniform chain but must not be silent no-ops either.
__device__ __forceinline__ void broadcastLimb0Body(const void* src, void* dst, const int idx, const int primeid) {
    if (ISU64(0)) {
        uint64_t in = ((const uint64_t*)src)[idx];
        SwitchModulus(in, 0, primeid);
        if (ISU64(primeid))
            ((uint64_t*)dst)[idx] = in;
        else
            ((uint32_t*)dst)[idx] = (uint32_t)in;
    } else {
        const uint32_t in32 = ((const uint32_t*)src)[idx];
        if (ISU64(primeid)) {
            uint64_t in = in32;
            SwitchModulus(in, 0, primeid);
            ((uint64_t*)dst)[idx] = in;
        } else {
            uint32_t in = in32;
            SwitchModulus(in, 0, primeid);
            ((uint32_t*)dst)[idx] = in;
        }
    }
}

__global__ void broadcastLimb0_(void** a) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = blockIdx.y + 1;
    broadcastLimb0Body(a[0], a[primeid], idx, primeid);
}

// COMPOSITESCALING ModRaise (see header). Grid: {N/threads, limbs}; src[k] holds a SNAPSHOT
// of source limb k's coefficients (raw device copy, width of prime k). All arithmetic is
// width-branched per prime; the accumulator uses the TARGET prime's width.
__global__ void compositeModRaise_(void** a, void** src, const __grid_constant__ int d, const uint64_t* qhatinv,
                                   const uint64_t* qhat) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = blockIdx.y;
    const int limbs = gridDim.y;

    if (ISU64(primeid)) {
        uint64_t acc = 0;
        for (int k = 0; k < d; ++k) {
            uint64_t x;
            if (ISU64(k)) {
                x = ((const uint64_t*)src[k])[idx];
                x = modmult<ALGO_BARRETT>(x, qhatinv[k], k);
            } else {
                uint32_t x32 = ((const uint32_t*)src[k])[idx];
                x32 = modmult<ALGO_BARRETT>(x32, (uint32_t)qhatinv[k], k);
                x = x32;
            }
            SwitchModulus(x, k, primeid);
            acc = modadd(acc, modmult<ALGO_BARRETT>(x, qhat[k * limbs + primeid], primeid), primeid);
        }
        ((uint64_t*)a[primeid])[idx] = acc;
    } else {
        uint32_t acc = 0;
        for (int k = 0; k < d; ++k) {
            uint32_t x;
            if (ISU64(k)) {
                // wide source, narrow target: switch modulus in 64-bit, then narrow
                uint64_t x64 = ((const uint64_t*)src[k])[idx];
                x64 = modmult<ALGO_BARRETT>(x64, qhatinv[k], k);
                SwitchModulus(x64, k, primeid);
                x = (uint32_t)x64;
            } else {
                x = ((const uint32_t*)src[k])[idx];
                x = modmult<ALGO_BARRETT>(x, (uint32_t)qhatinv[k], k);
                SwitchModulus(x, k, primeid);
            }
            acc = modadd(acc, modmult<ALGO_BARRETT>(x, (uint32_t)qhat[k * limbs + primeid], primeid), primeid);
        }
        ((uint32_t*)a[primeid])[idx] = acc;
    }
}

__global__ void broadcastLimb0_mgpu(void** a, const __grid_constant__ int primeid_init, void** limb0) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    broadcastLimb0Body(limb0[0], a[blockIdx.y], idx, primeid);
}

__global__ void copy_(void** a, void** b) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (ISU64(blockIdx.y)) {
        ((uint64_t*)b[blockIdx.y])[idx] = ((uint64_t*)a[blockIdx.y])[idx];
    } else {
        ((uint32_t*)b[blockIdx.y])[idx] = ((uint32_t*)a[blockIdx.y])[idx];
    }
}

/* Vectorized limb copy: FOUR elements per thread instead of one.
 *
 * copy_ moves one element per thread, so its warp count tracks the ELEMENT count, not the
 * byte count. A 32-bit chain carries ~2x the limbs of a 64-bit chain at the same logQ, so
 * for identical bytes it launches ~1.83x the warps — and a pure copy has no arithmetic to
 * hide the extra issue cost. Measured on Blackwell: copy_ costs 17.91 us/call on the n32
 * composite chain vs 12.72 on n64 (1.41x), while its arithmetic siblings add_/sub_ sit at
 * 1.07-1.16x. This kernel makes each thread move 16 B (uint4) or 32 B (ulonglong4), so the
 * copy is bandwidth-bound rather than issue-bound on both widths.
 *
 * Width selection is deliberately IDENTICAL to copy_ (ISU64(blockIdx.y)) so behaviour is
 * bit-for-bit unchanged. Grid must be {N/512, limbs} with 128 threads; the caller checks
 * N % 512 == 0 and falls back to copy_ otherwise (neither kernel takes a length argument,
 * so the grid must cover N exactly). Alignment holds: limb strides are N*4 / N*8 bytes,
 * both multiples of 32, on top of cudaMalloc's 256 B base alignment. */
__global__ void copy_v4_(void** a, void** b) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (ISU64(blockIdx.y)) {
        ((ulonglong4*)b[blockIdx.y])[i] = ((ulonglong4*)a[blockIdx.y])[i];
    } else {
        ((uint4*)b[blockIdx.y])[i] = ((uint4*)a[blockIdx.y])[i];
    }
}

/* TYPE-UNAWARE limb copy. A copy moves BYTES, so it has no business knowing whether the
 * limb holds u32 or u64 elements. Dropping the width branch removes a constant-memory load
 * (ISU64) and a branch per thread, makes mixed-width limbs correct by construction, and
 * sidesteps the latent ISU64(blockIdx.y) confusion in copy_/copy_v4_ (that macro wants a
 * PRIME ID; blockIdx.y is a limb SLOT — harmless only while both chains are width-uniform).
 *
 * It also unifies the tuning. Measured bytes/thread sweep on Blackwell (both chains peak at
 * 64 B and fall off monotonically past it — a memory-pipe property, not a chain property):
 *
 *     bytes/thread   16     32     64      128     256
 *     n32 GB/s      1204   1205   1240    1163    1053
 *     n64 GB/s        -    1267   1302    1150    1064
 *
 * OPS=4 is 64 B/thread on BOTH chains with identical code; only the grid differs, through
 * bytes-per-limb, which the host already knows. Type-unaware at 64 B measured 1243 GB/s on
 * n32 and 1312-1318 on n64 — equal or better than the typed kernel everywhere, and +3.2%
 * (n32) / +3.6-4.0% (n64) over the shipped copy_v4_ (which sits at 16 B and 32 B/thread
 * respectively, i.e. below the knee on both).
 *
 * Grid must be {bytes_per_limb/(16*OPS*128), nlimbs}, block 128 — the kernel carries no
 * length, so the grid has to cover the limb exactly. */
template <int OPS>
__global__ void copy_bytes_(void** a, void** b) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
#pragma unroll
    for (int q = 0; q < OPS; ++q)
        ((uint4*)b[blockIdx.y])[OPS * i + q] = ((uint4*)a[blockIdx.y])[OPS * i + q];
}

/* Cross-TU launcher. A __global__ TEMPLATE launched from a TU that only sees its declaration
 * gets a weak local stub with no device code in that TU's fatbin => 'invalid device function'
 * (documented for the dot kernels above, job 50426241). Keep every instantiation here. */
void launchCopyBytes(dim3 grid, dim3 block, cudaStream_t stream, void** a, void** b, int ops) {
    switch (ops) {
        case 1: copy_bytes_<1><<<grid, block, 0, stream>>>(a, b); break;
        case 2: copy_bytes_<2><<<grid, block, 0, stream>>>(a, b); break;
        case 4: copy_bytes_<4><<<grid, block, 0, stream>>>(a, b); break;
        default: throw std::runtime_error("launchCopyBytes: unsupported ops (expect 1, 2 or 4)");
    }
}

__global__ void copy1D_(void* a, void* b) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    ((uint64_t*)b)[idx] = ((uint64_t*)a)[idx];
}

template <ALGO algo>
__global__ void Scalar_mult_(void** a, const uint64_t* b, const __grid_constant__ int primeid_init,
                             const uint64_t* shoup_mu) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];

    if (ISU64(primeid)) {
        ((uint64_t*)a[blockIdx.y])[idx] =
            modmult<algo>(((uint64_t*)a[blockIdx.y])[idx], b[primeid], primeid, shoup_mu ? shoup_mu[primeid] : 0);
    } else {
        ((uint32_t*)a[blockIdx.y])[idx] = modmult<algo>(((uint32_t*)a[blockIdx.y])[idx], (uint32_t)b[primeid], primeid,
                                                        (uint32_t)(shoup_mu ? shoup_mu[primeid] : 0));
    }
}

__global__ void eval_linear_w_sum_(const __grid_constant__ int n, void** a, void*** bs, uint64_t* w,
                                   const __grid_constant__ int primeid_init) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    constexpr ALGO algo = ALGO_BARRETT;

    if (ISU64(primeid)) {
        uint64_t res = modmult<algo>(((uint64_t*)(bs[0])[blockIdx.y])[idx], w[primeid], primeid);
        for (int i = 1; i < n; ++i) {
            uint64_t temp = modmult<algo>(((uint64_t*)(bs[i])[blockIdx.y])[idx], w[i * MAXP + primeid], primeid);
            res = modadd(res, temp, primeid);
        }
        ((uint64_t*)a[blockIdx.y])[idx] = res;
    } else {
        uint32_t res = modmult<algo>(((uint32_t*)(bs[0])[blockIdx.y])[idx], (uint32_t)w[primeid], primeid);
        for (int i = 1; i < n; ++i) {
            uint32_t temp = modmult<algo>(((uint32_t*)(bs[i])[blockIdx.y])[idx], (uint32_t)(w[i * MAXP + primeid]), primeid);
            res = modadd(res, temp, primeid);
        }
        ((uint32_t*)a[blockIdx.y])[idx] = res;
    }
}

// Lever 1b-i: funnelshift extraction of coefficient idx from a bits-per-coefficient packed
// stream. Reads two overlapping 32-bit words; warp neighbors overlap so the extra word is
// L1-served and DRAM sees ~bits/32 of the dense stream. The producer (packKsk_) zeroes one
// guard word past the stream so the idx==N-1 speculative q[1] read is always in-bounds.
__device__ __forceinline__ uint32_t kskUnpack(const void* p, const uint32_t idx, const uint32_t bits,
                                              const uint32_t mask) {
    const uint32_t bitoff = idx * bits;
    const uint32_t* q = (const uint32_t*)p + (bitoff >> 5);
    return __funnelshift_r(q[0], q[1], bitoff & 31) & mask;
}

__global__ void expandKskA_(uint32_t* out, const KskSeedWords seed, const int digit, const uint32_t p,
                            const uint32_t n16, const int N) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
        out[idx] = kskexpand::expand_coeff(seed.k, (uint32_t)digit, p, (uint32_t)idx, n16);
}

__global__ void packKsk_(uint32_t* out, const uint32_t* in, const int N, const int bits) {
    const uint32_t total = (uint32_t)(((uint64_t)N * bits + 31) >> 5);
    const uint32_t w = threadIdx.x + blockIdx.x * blockDim.x;
    if (w > total)
        return;
    if (w == total) {  // zeroed guard word for the consumer funnelshift
        out[w] = 0;
        return;
    }
    const uint32_t mask = (1u << bits) - 1u;
    const uint64_t bit0 = (uint64_t)w << 5;
    uint32_t k = (uint32_t)(bit0 / bits);
    const uint32_t off = (uint32_t)(bit0 - (uint64_t)k * bits);
    uint64_t acc = (uint64_t)(in[k] & mask) >> off;
    for (uint32_t filled = bits - off; filled < 32 && k + 1 < (uint32_t)N; filled += bits)
        acc |= (uint64_t)(in[++k] & mask) << filled;
    out[w] = (uint32_t)acc;
}

// Lever 1 (2026-07-27, ncu job 50417213): these dot kernels are REGISTER-occupancy-limited,
// not DRAM-bound — occupancy_limit_registers=8 blocks/SM (>40 regs/thread) ⇒ only ~49% warps
// active, DRAM at 41-47% of peak, SM ~50%. __launch_bounds__(128, 12) caps regs at ~42 and
// raises residency to 12 blocks/SM (+50% warps) so the streamed KSK reads have latency cover.
// Gated by wrapper bitcmp (numerics untouched) + wall A/B; revert if spills outweigh it.
// Lever 1b-i: KSK_BITS is the COMPILE-TIME packed width (0 = dense). A runtime-bits variant
// measured +2.4% (gate 50427306 / ncu 50428101): the extra bits/mask registers dropped the
// packed arm 16 -> 12 blocks/SM (warps 93.8 -> 73%, DRAM 72 -> 52%). Compile-time width makes
// the mask/shift immediates, and packed arms pin min-blocks 16 to hold the occupancy tier.
template <int KSK_BITS>
__global__ void __launch_bounds__(128, KSK_BITS ? 16 : 12)
    fusedDotKSK_2_(void** out1, void** sout1, void** out2, void** sout2, void*** digits, int num_d, int id,
                   int num_special, int init) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;

    const int blky = blockIdx.y + init;
    //num_special = C_.K;
    int primeid;
    if (blky < num_special) {
        primeid = C_.primeid_digit_to[0][blky];
    } else {
        primeid = C_.primeid_partition[id][blky - num_special];
    }

    const int primeid_digit = C_.primeid_digit[primeid];

    int pos_dec = blky - num_special;

    if (C_.type == 0) {
        // n32 (Lever 1 fix 2): all-U32 chain fast path — the generic arm below promotes every
        // operand to uint64_t and runs 64-bit Barrett per term on 27-bit primes. 32-bit Barrett
        // (Neal_mult_32) + u32 accumulators compute the IDENTICAL canonical residues (same
        // reduce-then-add order) in ~1/3 the instructions AND fewer registers — registers are
        // the measured occupancy limiter of this kernel (ncu 50417213). Grid-uniform branch.
        uint32_t a1, a2;
        for (int i = 0; i < num_d; ++i) {
            const bool decomp = (i == primeid_digit);
            const int pos = C_.pos_in_digit[i][primeid];
            const int p = decomp ? pos_dec : pos;
            const uint32_t in = ((uint32_t*)digits[i + decomp * 3 * C_.dnum][p])[idx];
            uint32_t kska, kskb;
            if constexpr (KSK_BITS) {
                kska = kskUnpack(digits[C_.dnum + i + decomp * 3 * C_.dnum][p], idx, KSK_BITS,
                                 (1u << KSK_BITS) - 1u);
                kskb = kskUnpack(digits[2 * C_.dnum + i + decomp * 3 * C_.dnum][p], idx, KSK_BITS,
                                 (1u << KSK_BITS) - 1u);
            } else {
                kska = ((uint32_t*)digits[C_.dnum + i + decomp * 3 * C_.dnum][p])[idx];
                kskb = ((uint32_t*)digits[2 * C_.dnum + i + decomp * 3 * C_.dnum][p])[idx];
            }
            const uint32_t m1 = modmult<ALGO_BARRETT>(in, kska, primeid);
            const uint32_t m2 = modmult<ALGO_BARRETT>(in, kskb, primeid);
            if (i == 0) {
                a1 = m1;
                a2 = m2;
            } else {
                a1 = modadd(a1, m1, primeid);
                a2 = modadd(a2, m2, primeid);
            }
        }
        if (primeid < C_.L) {
            ((uint32_t*)out1[pos_dec])[idx] = a1;
            ((uint32_t*)out2[pos_dec])[idx] = a2;
        } else {
            ((uint32_t*)sout1[primeid - C_.L])[idx] = a1;
            ((uint32_t*)sout2[primeid - C_.L])[idx] = a2;
        }
        return;
    }

    /*
    int i = 0;
    bool decomp = (i == primeid_digit);
    int pos = C_.pos_in_digit[i][primeid];

    uint64_t aux1, aux2;

    uint64_t in = ((uint64_t*)digits[0 + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];
    aux1 = modmult<ALGO_BARRETT>(in, ((uint64_t*)digits[C_.dnum + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx],
                                 primeid);
    aux2 = modmult<ALGO_BARRETT>(
        in, ((uint64_t*)digits[2 * C_.dnum + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx], primeid);

    for (i = 1; i < num_d; ++i) {
        decomp = (i == primeid_digit);
        pos = C_.pos_in_digit[i][primeid];
        in = ((uint64_t*)digits[i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];
        uint64_t add1 = modmult<ALGO_BARRETT>(
            in, ((uint64_t*)digits[C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx], primeid);
        uint64_t add2 = modmult<ALGO_BARRETT>(
            in, ((uint64_t*)digits[2 * C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx], primeid);
        aux1 = modadd(aux1, add1, primeid);
        aux2 = modadd(aux2, add2, primeid);
    }
    */
    uint64_t aux1, aux2;

    for (int i = 0; i < num_d; ++i) {
        bool decomp = (i == primeid_digit);
        int pos = C_.pos_in_digit[i][primeid];

        //printf("Digit %d: in: %p\n", i, digits);
        //printf("Digit %d: in: %p, kska: %p, kskb: %p\n", i, digits[i + decomp * 3 * C_.dnum],
        //       digits[C_.dnum + i + decomp * 3 * C_.dnum], digits[2 * C_.dnum + i + decomp * 3 * C_.dnum]);

        uint64_t in;
        if (ISU64(primeid)) {
            in = ((uint64_t*)digits[i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];
        } else {
            in = ((uint32_t*)digits[i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];
        }
        uint64_t add1;
        uint64_t add2;
        if (ISU64(primeid)) {
            add1 = modmult<ALGO_BARRETT>(
                in, ((uint64_t*)digits[C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx], primeid);
            add2 = modmult<ALGO_BARRETT>(
                in, ((uint64_t*)digits[2 * C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx], primeid);
        } else {
            add1 = modmult<ALGO_BARRETT>(
                in, (uint64_t)((uint32_t*)digits[C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx], primeid);
            add2 = modmult<ALGO_BARRETT>(
                in, (uint64_t)((uint32_t*)digits[2 * C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx], primeid);
        }

        if (i == 0) {
            aux1 = add1;
            aux2 = add2;
        } else {
            aux1 = modadd(aux1, add1, primeid);
            aux2 = modadd(aux2, add2, primeid);
        }
    }

    if (primeid < C_.L) {
        if (ISU64(primeid)) {
            ((uint64_t*)out1[pos_dec])[idx] = aux1;
            ((uint64_t*)out2[pos_dec])[idx] = aux2;
        } else {
            ((uint32_t*)out1[pos_dec])[idx] = (uint32_t)aux1;
            ((uint32_t*)out2[pos_dec])[idx] = (uint32_t)aux2;
        }
    } else {
        if (ISU64(primeid)) {
            ((uint64_t*)sout1[primeid - C_.L])[idx] = aux1;
            ((uint64_t*)sout2[primeid - C_.L])[idx] = aux2;
        } else {
            ((uint32_t*)sout1[primeid - C_.L])[idx] = (uint32_t)aux1;
            ((uint32_t*)sout2[primeid - C_.L])[idx] = (uint32_t)aux2;
        }
    }
}

// Same-TU launcher (see the .cuh note: cross-TU template-kernel launches hit
// 'invalid device function' — the launch must live in the defining TU). Supported packed
// widths are the instantiated set {27, 28}; kskPackBitsPolicy only arms those.
void launchFusedDotKSK_2(dim3 grid, dim3 block, cudaStream_t stream, void** out1, void** sout1, void** out2,
                         void** sout2, void*** digits, int num_d, int id, int num_special, int init,
                         int ksk_pack_bits) {
    switch (ksk_pack_bits) {
        case 0:
            fusedDotKSK_2_<0><<<grid, block, 0, stream>>>(out1, sout1, out2, sout2, digits, num_d, id, num_special,
                                                          init);
            break;
        case 27:
            fusedDotKSK_2_<27><<<grid, block, 0, stream>>>(out1, sout1, out2, sout2, digits, num_d, id, num_special,
                                                           init);
            break;
        case 28:
            fusedDotKSK_2_<28><<<grid, block, 0, stream>>>(out1, sout1, out2, sout2, digits, num_d, id, num_special,
                                                           init);
            break;
        default:
            throw std::runtime_error("launchFusedDotKSK_2: unsupported ksk_pack_bits " +
                                     std::to_string(ksk_pack_bits));
    }
}

constexpr bool PRINT = false;

// Lever 1: same register-occupancy treatment as fusedDotKSK_2_ above (ncu 50417213).
// Lever 1b-i: same compile-time KSK_BITS treatment as fusedDotKSK_2_ above (ncu 50428101).
template <int KSK_BITS>
__global__ void __launch_bounds__(128, KSK_BITS ? 16 : 12)
    hoistedRotateDotKSK_2_(void*** din1, void** c0, void*** out1, void*** sout1, void*** out2,
                           void*** sout2, const int n, const int* indexes, void*** digits, int num_d,
                           int id, int num_special, int init, void** sc0, bool c0_modup) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int blky = blockIdx.y + init;

    const int primeid =
        (blky < num_special) ? C_.primeid_digit_to[0][blky] : C_.primeid_partition[id][blky - num_special];

    const int primeid_digit = C_.primeid_digit[primeid];
    const int pos_dec = blky - num_special;

    extern __shared__ char buffer[];

    if (C_.type == 0) {
        // n32 (Lever 1 fix 2): u32 fast path — see fusedDotKSK_2_ above. The shared digit
        // cache is reinterpreted as u32 (uses half the allocation; layout self-consistent
        // within this arm); residues and store order identical to the generic arm => bit-exact.
        uint32_t* in1s = ((uint32_t*)buffer) + num_d * threadIdx.x;
        for (int i = 0; i < num_d; ++i) {
            const bool decomp = (i == primeid_digit);
            const int pos = C_.pos_in_digit[i][primeid];
            in1s[i] = ((uint32_t*)din1[i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];
        }
        uint32_t in2 = 0;
        if (c0_modup || primeid < C_.L) {
            in2 = primeid < C_.L ? ((uint32_t*)c0[pos_dec])[idx] : ((uint32_t*)sc0[primeid - C_.L])[idx];
            if (!c0_modup && primeid < C_.L)
                in2 = modmult<ALGO_SHOUP>(in2, (uint32_t)C_.P[primeid], primeid, (uint32_t)C_.P_shoup[primeid]);
        }
        for (int j = 0; j < n; ++j) {
            const int offset = j * 3 * 2 * C_.dnum;
            uint32_t aux1, aux2;
            for (int i = 0; i < num_d; ++i) {
                const bool decomp = (i == primeid_digit);
                const int pos = C_.pos_in_digit[i][primeid];
                const int p = decomp ? pos_dec : pos;
                uint32_t kska, kskb;
                if constexpr (KSK_BITS) {
                    kska = kskUnpack(digits[offset + C_.dnum + i + decomp * 3 * C_.dnum][p], idx, KSK_BITS,
                                     (1u << KSK_BITS) - 1u);
                    kskb = kskUnpack(digits[offset + 2 * C_.dnum + i + decomp * 3 * C_.dnum][p], idx, KSK_BITS,
                                     (1u << KSK_BITS) - 1u);
                } else {
                    kska = ((uint32_t*)digits[offset + C_.dnum + i + decomp * 3 * C_.dnum][p])[idx];
                    kskb = ((uint32_t*)digits[offset + 2 * C_.dnum + i + decomp * 3 * C_.dnum][p])[idx];
                }
                const uint32_t add1 = modmult<ALGO_BARRETT>(in1s[i], kska, primeid);
                const uint32_t add2 = modmult<ALGO_BARRETT>(in1s[i], kskb, primeid);
                if (i == 0) {
                    aux1 = add1;
                    aux2 = (c0_modup || primeid < C_.L) ? modadd(in2, add2, primeid) : add2;
                } else {
                    aux1 = modadd(aux1, add1, primeid);
                    aux2 = modadd(aux2, add2, primeid);
                }
            }
            const uint32_t out_idx = automorph_slot(C_.logN, indexes[j], idx);
            if (primeid < C_.L) {
                ((uint32_t*)out1[j][pos_dec])[out_idx] = aux1;
                ((uint32_t*)out2[j][pos_dec])[out_idx] = aux2;
            } else {
                ((uint32_t*)sout1[j][primeid - C_.L])[out_idx] = aux1;
                ((uint32_t*)sout2[j][primeid - C_.L])[out_idx] = aux2;
            }
        }
        return;
    }

    uint64_t* in1 = ((uint64_t*)buffer) + num_d * threadIdx.x;

    for (int i = 0; i < num_d; ++i) {
        bool decomp = (i == primeid_digit);
        int pos = C_.pos_in_digit[i][primeid];
        if (ISU64(primeid)) {
            in1[i] = ((uint64_t*)din1[i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];
        } else {
            in1[i] = ((uint32_t*)din1[i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];
        }

        if (PRINT && idx == 0 && blky == 6)
            printf("In %d : %lu\n", i, in1[i]);
    }

    uint64_t in2 = 0;

    if (c0_modup || primeid < C_.L) {
        if (ISU64(primeid)) {
            in2 = primeid < C_.L ? ((uint64_t*)c0[pos_dec])[idx] : ((uint64_t*)sc0[primeid - C_.L])[idx];
            if (!c0_modup && primeid < C_.L)
                in2 = modmult<ALGO_SHOUP>(in2, C_.P[primeid], primeid, C_.P_shoup[primeid]);
        } else {
            in2 = primeid < C_.L ? ((uint32_t*)c0[pos_dec])[idx] : ((uint32_t*)sc0[primeid - C_.L])[idx];
            if (!c0_modup && primeid < C_.L)
                in2 = modmult<ALGO_SHOUP>((uint32_t)in2, (uint32_t)C_.P[primeid], primeid,
                                          (uint32_t)C_.P_shoup[primeid]);
        }
    }

    if (PRINT && idx == 0 && blky == 6 && threadIdx.y == 0)
        printf("In c0: %lu\n", in2);

    for (int j = 0; j < n; ++j) {
        uint64_t aux1, aux2;
        int offset = j * 3 * 2 * C_.dnum;

        for (int i = 0; i < num_d; ++i) {
            bool decomp = (i == primeid_digit);
            int pos = C_.pos_in_digit[i][primeid];
            uint64_t kska, kskb;
            if (ISU64(primeid)) {
                kska = ((uint64_t*)digits[offset + C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];
                kskb =
                    ((uint64_t*)digits[offset + 2 * C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];
            } else {
                kska = ((uint32_t*)digits[offset + C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];
                kskb =
                    ((uint32_t*)digits[offset + 2 * C_.dnum + i + decomp * 3 * C_.dnum][decomp ? pos_dec : pos])[idx];
            }
            uint64_t add1 = modmult<ALGO_BARRETT>(in1[i], kska, primeid);

            if (PRINT && idx == 0 && blky == 6)
                printf("kska %d : %lu\n", i, kska);

            if (PRINT && idx == 0 && blky == 6)
                printf("add1 %d : %lu\n", i, add1);
            uint64_t add2 = modmult<ALGO_BARRETT>(in1[i], kskb, primeid);

            if (PRINT && idx == 0 && blky == 6)
                printf("kskb %d : %lu\n", i, kskb);
            if (PRINT && idx == 0 && blky == 6)
                printf("add2 %d : %lu\n", i, add2);

            if (i == 0) {
                aux1 = add1;
                if (c0_modup || primeid < C_.L) {
                    aux2 = modadd(in2, add2, primeid);
                } else {
                    aux2 = add2;
                }
            } else {
                aux1 = modadd(aux1, add1, primeid);
                aux2 = modadd(aux2, add2, primeid);
            }
            if (PRINT && idx == 0 && blky == 6)
                printf("aux1 %d : %lu\n", i, aux1);
            if (PRINT && idx == 0 && blky == 6)
                printf("aux2 %d : %lu\n", i, aux2);
        }

        if (PRINT && idx == 0 && blky == 6)
            printf("%d : %d %d %d %lu\n", j, indexes[j], C_.logN, automorph_slot(C_.logN, indexes[j], idx), aux1);
        if (PRINT && idx == 0 && blky == 6)
            printf("%d : %d %d %d %lu\n", j, indexes[j], C_.logN, automorph_slot(C_.logN, indexes[j], idx), aux2);

        uint32_t out_idx = automorph_slot(C_.logN, indexes[j], idx);
        //uint32_t out_idx = idx;
        if (primeid < C_.L) {
            if (ISU64(primeid)) {
                ((uint64_t*)out1[j][pos_dec])[out_idx] = aux1;
                ((uint64_t*)out2[j][pos_dec])[out_idx] = aux2;
            } else {
                ((uint32_t*)out1[j][pos_dec])[out_idx] = (uint32_t)aux1;
                ((uint32_t*)out2[j][pos_dec])[out_idx] = (uint32_t)aux2;
            }
        } else {
            if (ISU64(primeid)) {
                ((uint64_t*)sout1[j][primeid - C_.L])[out_idx] = aux1;
                ((uint64_t*)sout2[j][primeid - C_.L])[out_idx] = aux2;
            } else {
                ((uint32_t*)sout1[j][primeid - C_.L])[out_idx] = (uint32_t)aux1;
                ((uint32_t*)sout2[j][primeid - C_.L])[out_idx] = (uint32_t)aux2;
            }
        }
    }
}

void launchHoistedRotateDotKSK_2(dim3 grid, dim3 block, size_t shmem, cudaStream_t stream, void*** din1, void** c0,
                                 void*** out1, void*** sout1, void*** out2, void*** sout2, int n, const int* indexes,
                                 void*** digits, int num_d, int id, int num_special, int init, void** sc0,
                                 bool c0_modup, int ksk_pack_bits) {
    switch (ksk_pack_bits) {
        case 0:
            hoistedRotateDotKSK_2_<0><<<grid, block, shmem, stream>>>(din1, c0, out1, sout1, out2, sout2, n, indexes,
                                                                      digits, num_d, id, num_special, init, sc0,
                                                                      c0_modup);
            break;
        case 27:
            hoistedRotateDotKSK_2_<27><<<grid, block, shmem, stream>>>(din1, c0, out1, sout1, out2, sout2, n, indexes,
                                                                       digits, num_d, id, num_special, init, sc0,
                                                                       c0_modup);
            break;
        case 28:
            hoistedRotateDotKSK_2_<28><<<grid, block, shmem, stream>>>(din1, c0, out1, sout1, out2, sout2, n, indexes,
                                                                       digits, num_d, id, num_special, init, sc0,
                                                                       c0_modup);
            break;
        default:
            throw std::runtime_error("launchHoistedRotateDotKSK_2: unsupported ksk_pack_bits " +
                                     std::to_string(ksk_pack_bits));
    }
}

__global__ void hoistedRotateDotKSKBatched___(void*** c1, void*** din1, void*** c0, void*** sc0, void*** out1,
                                              void*** sout1, void*** out2, void*** sout2, const int n,
                                              const int* indexes, void*** digits, int num_d, int id, int num_special,
                                              int init_, bool c0_modup) {
    // cg::thread_block tb = cg::this_thread_block();
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int blky = blockIdx.y + init_;

    const int primeid =
        (blky < num_special) ? C_.primeid_digit_to[0][blky] : C_.primeid_partition[id][blky - num_special];

    const int primeid_digit = C_.primeid_digit[primeid];
    const int pos_dec = blky - num_special;
    //__shared__ cuda::barrier<cuda::thread_scope_block> bar;
    extern __shared__ char buffer[];

    // Initialize barrier (single thread)
    //if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    //    init(&bar, blockDim.x * blockDim.y * blockDim.z);
    //}
    //__syncthreads();

    int stride_in = blockDim.x * blockDim.z;
    int stride_ksk = blockDim.x * blockDim.y;
    // uint64_t* in_base = ((uint64_t*)buffer);
    // uint64_t* ksk_base = ((uint64_t*)buffer) + num_d * stride_in;
    uint64_t* in1 = ((uint64_t*)buffer) + (threadIdx.x + blockDim.x * threadIdx.z);
    uint64_t* ksk = ((uint64_t*)buffer) + num_d * stride_in + (threadIdx.y * blockDim.x + threadIdx.x);

    if (PRINT && idx == 0 && blky == 6 && threadIdx.y == 0)
        printf("in1: %p, ksk: %p\n", in1, ksk);

    //int n_elem = blockDim.x;
    /*
    for (int z = 0; z < blockDim.z; z++) {
        for (uint32_t i = 0; i < num_d; ++i) {
            bool decomp = (i == primeid_digit);
            int pos = C_.pos_in_digit[i][primeid];
            cuda::memcpy_async(
                tb, in_base + n_elem * z + stride_in * i,
                ((uint64_t*)(decomp ? c1[z] : din1[num_d * z + i])[decomp ? pos_dec : pos]) + blockIdx.x * blockDim.x,
                cuda::aligned_size_t<16>(n_elem * sizeof(uint64_t)), bar);
        }
    }
*/
    /*
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (uint32_t i = 0; i < num_d; ++i) {
            bool decomp = (i == primeid_digit);
            int pos = C_.pos_in_digit[i][primeid];
            cuda::memcpy_async(
                in_base + n_elem * threadIdx.z + stride_in * i,
                ((uint64_t*)(decomp ? c1[threadIdx.z] : din1[num_d * threadIdx.z + i])[decomp ? pos_dec : pos]) +
                    blockIdx.x * blockDim.x,
                cuda::aligned_size_t<16>(n_elem * sizeof(uint64_t)), bar);
        }
    }
*/

    for (uint32_t i = threadIdx.y; i < num_d; i += blockDim.y) {
        bool decomp = (i == primeid_digit);
        int pos = C_.pos_in_digit[i][primeid];

        in1[i * stride_in] =
            ((uint64_t*)(decomp ? c1[threadIdx.z] : din1[num_d * threadIdx.z + i])[decomp ? pos_dec : pos])[idx];

        if (PRINT && idx == 0 && blky == 6)
            printf("In %d : %lu\n", i, in1[i * stride_in]);
    }
    __syncthreads();
    uint64_t in2 = 0;
    if ((c0_modup && threadIdx.y == 1) || primeid < C_.L) {
        in2 = threadIdx.y == 1 ? ((primeid < C_.L) ? ((uint64_t*)c0[threadIdx.z][pos_dec])[idx]
                                                   : ((uint64_t*)sc0[threadIdx.z][primeid - C_.L])[idx])
                               : in1[stride_in * primeid_digit];
        if (!c0_modup || threadIdx.y == 0)
            in2 = modmult<ALGO_SHOUP>(in2, C_.P[primeid], primeid, C_.P_shoup[primeid]);
    }

    if (PRINT && idx == 0 && blky == 6 && threadIdx.y == 1)
        printf("In c0: %lu\n", in2);

    for (int j = 0; j < n; ++j) {

        if (j > 0)
            __syncthreads();

        if (digits[(2 * j + threadIdx.y) * (num_d + 1)]) {
            uint64_t aux;
            for (int i = threadIdx.z; i < num_d; i += blockDim.z) {
                bool decomp = (i == primeid_digit);
                int pos = C_.pos_in_digit[i][primeid];
                /*
            if (threadIdx.x == 0) {


                cuda::memcpy_async(
                    ksk_base + n_elem * threadIdx.y + stride_ksk * i,
                    ((uint64_t*)
                         digits[(2 * j + threadIdx.y) * (num_d + 1) + (decomp ? num_d : i)][decomp ? pos_dec : pos]) +
                        blockIdx.x * blockDim.x,
                    cuda::aligned_size_t<16>(n_elem * sizeof(uint64_t)), bar);
            }*/
                uint64_t** ksk_from = (uint64_t**)digits[(2 * j + threadIdx.y) * (num_d + 1) + (decomp ? num_d : i)];

                ksk[i * stride_ksk] = ksk_from[decomp ? pos_dec : pos][idx];

                if (PRINT && idx == 0 && blky == 6)
                    printf("ksk %d : %lu\n", i, ksk[i * stride_ksk]);
            }
            //bar.arrive_and_wait();
            __syncthreads();

            for (int i = 0; i < num_d; ++i) {
                uint64_t add = modmult<ALGO_BARRETT>(in1[i * stride_in], ksk[i * stride_ksk], primeid);

                if (PRINT && idx == 0 && blky == 6)
                    printf("add %d : %lu\n", i, add);
                if (i == 0) {
                    if ((c0_modup || primeid < C_.L) && threadIdx.y == 1) {
                        aux = modadd(in2, add, primeid);
                    } else {
                        aux = add;
                    }
                } else {
                    aux = modadd(aux, add, primeid);
                }
                if (PRINT && idx == 0 && blky == 6)
                    printf("aux %d : %lu\n", i, aux);
            }

            if (PRINT && idx == 0 && blky == 6)
                printf("%d : %d %d %d %lu\n", j, indexes[j], C_.logN, automorph_slot(C_.logN, indexes[j], idx), aux);

            uint32_t out_idx = automorph_slot(C_.logN, indexes[j], idx);
            //uint32_t out_idx = idx;
            uint64_t* out = (primeid < C_.L)
                                ? (threadIdx.y == 0 ? (uint64_t*)out1[threadIdx.z * n + j][pos_dec]
                                                    : (uint64_t*)out2[threadIdx.z * n + j][pos_dec])
                                : (threadIdx.y == 0 ? (uint64_t*)sout1[threadIdx.z * n + j][primeid - C_.L]
                                                    : (uint64_t*)sout2[threadIdx.z * n + j][primeid - C_.L]);

            out[out_idx] = aux;
        } else {

            uint64_t* out = (primeid < C_.L)
                                ? (threadIdx.y == 0 ? (uint64_t*)out1[threadIdx.z * n + j][pos_dec]
                                                    : (uint64_t*)out2[threadIdx.z * n + j][pos_dec])
                                : (threadIdx.y == 0 ? (uint64_t*)sout1[threadIdx.z * n + j][primeid - C_.L]
                                                    : (uint64_t*)sout2[threadIdx.z * n + j][primeid - C_.L]);

            if (PRINT && idx == 0)
                printf("y: %d Primeid_digit: %d, in2: %lu %lu\n", blky, primeid_digit, in2,
                       in1[stride_in * (primeid_digit + (primeid_digit == -1))]);

            out[idx] = in2;
        }
    }
}

/*
__global__ void hoistedRotateDotKSKBatched___(void*** c1, void*** din1, void*** c0, void*** sc0, void*** out1,
                                              void*** sout1, void*** out2, void*** sout2, const int n,
                                              const int* indexes, void*** digits, int num_d, int id, int num_special,
                                              int init, bool c0_modup) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int blky = blockIdx.y + init;

    const int primeid =
        (blky < num_special) ? C_.primeid_digit_to[0][blky] : C_.primeid_partition[id][blky - num_special];

    const int primeid_digit = C_.primeid_digit[primeid];
    const int pos_dec = blky - num_special;

    extern __shared__ char buffer[];

    int stride_in = blockDim.x * blockDim.z;
    int stride_ksk = blockDim.x * blockDim.y;
    uint64_t* in1 = ((uint64_t*)buffer) + (threadIdx.x + blockDim.x * threadIdx.z);
    uint64_t* ksk = ((uint64_t*)buffer) + num_d * stride_in + (threadIdx.x * blockDim.y + threadIdx.y);

    if (PRINT && idx == 0 && blky == 6 && threadIdx.y == 0)
        printf("in1: %p, ksk: %p\n", in1, ksk);

    for (int i = threadIdx.y; i < num_d; i += blockDim.y) {
        bool decomp = (i == primeid_digit);
        int pos = C_.pos_in_digit[i][primeid];
        in1[i * stride_in] =
            ((uint64_t*)(decomp ? c1[threadIdx.z] : din1[num_d * threadIdx.z + i])[decomp ? pos_dec : pos])[idx];

        if (PRINT && idx == 0 && blky == 6)
            printf("In %d : %lu\n", i, in1[i * stride_in]);
    }

    uint64_t in2 = 0;
    if ((c0_modup || primeid < C_.L) && threadIdx.y == 1) {
        in2 = ((uint64_t*)(primeid < C_.L ? c0 : sc0)[threadIdx.z][pos_dec])[idx];
        if (!c0_modup)
            in2 = modmult<ALGO_SHOUP>(in2, C_.P[primeid], primeid, C_.P_shoup[primeid]);
    }

    if (PRINT && idx == 0 && blky == 6 && threadIdx.y == 1)
        printf("In c0: %lu\n", in2);

    for (int j = 0; j < n; ++j) {
        uint64_t aux;

        if (j > 0)
            __syncthreads();
        for (int i = threadIdx.z; i < num_d; i += blockDim.z) {
            bool decomp = (i == primeid_digit);
            int pos = C_.pos_in_digit[i][primeid];
            ksk[i * stride_ksk] =
                ((uint64_t*)
                     digits[(2 * j + threadIdx.y) * (num_d + 1) + (decomp ? num_d : i)][decomp ? pos_dec : pos])[idx];

            if (PRINT && idx == 0 && blky == 6)
                printf("ksk %d : %lu\n", i, ksk[i * stride_ksk]);
        }
        __syncthreads();
        for (int i = 0; i < num_d; ++i) {
            uint64_t add = modmult<ALGO_BARRETT>(in1[i * stride_in], ksk[i * stride_ksk], primeid);

            if (PRINT && idx == 0 && blky == 6)
                printf("add %d : %lu\n", i, add);
            if (i == 0) {
                if ((c0_modup || primeid < C_.L) && threadIdx.y == 1) {
                    aux = modadd(in2, add, primeid);
                } else {
                    aux = add;
                }
            } else {
                aux = modadd(aux, add, primeid);
            }
            if (PRINT && idx == 0 && blky == 6)
                printf("aux %d : %lu\n", i, aux);
        }

        if (PRINT && idx == 0 && blky == 6)
            printf("%d : %d %d %d %lu\n", j, indexes[j], C_.logN, automorph_slot(C_.logN, indexes[j], idx), aux);

        uint32_t out_idx = automorph_slot(C_.logN, indexes[j], idx);
        //uint32_t out_idx = idx;
        uint64_t* out = (primeid < C_.L) ? (threadIdx.y == 0 ? (uint64_t*)out1[threadIdx.z * n + j][pos_dec]
                                                             : (uint64_t*)out2[threadIdx.z * n + j][pos_dec])
                                         : (threadIdx.y == 0 ? (uint64_t*)sout1[threadIdx.z * n + j][primeid - C_.L]
                                                             : (uint64_t*)sout2[threadIdx.z * n + j][primeid - C_.L]);

        out[out_idx] = aux;
    }
}
*/

// n32: the gStep > 8 sibling of dotProductLtBatchedPt3___, used by the non-batched linear
// transform (LimbPartition::dotProductPt) and by CoeffsToSlots' wide steps -- at logN=16 a
// single bootstrap exercises BOTH kernels. It was uint64_t-hardcoded with no width branch,
// so on U32 limbs it read/wrote at twice the element size. Same treatment: templated body,
// width branch at the top. modmult<algo> and modadd both have uint32_t overloads.
template <typename T>
__device__ __forceinline__ void dotProductPtBody(void** c0, void** c1, void*** data, const size_t ptroffset,
                                                 const int n, const int idx, const int primeid) {
    constexpr ALGO algo = ALGO_BARRETT;

    T out0, out1;
    T in = ((T*)data[n * 2][ptroffset + blockIdx.y])[idx];
    if (PRINT && idx == 0 && blockIdx.y == 0)
        printf("LT:, b: %d, g:, in pt: %lu \n", 0, (unsigned long)in);
    out0 = modmult<algo>(in, ((T*)data[0][ptroffset + blockIdx.y])[idx], primeid);

    if (PRINT && idx == 0 && blockIdx.y == 0)
        printf("LT: %d, b: %d, in c0: %lu \n", -1, 0, (unsigned long)((T*)data[0][ptroffset + blockIdx.y])[idx]);

    out1 = modmult<algo>(in, ((T*)data[n][ptroffset + blockIdx.y])[idx], primeid);

    if (PRINT && idx == 0 && blockIdx.y == 0)
        printf("LT: %d, b: %d, in c1: %lu \n", -1, 0, (unsigned long)((T*)data[n][ptroffset + blockIdx.y])[idx]);

    for (int i = 1; i < n; ++i) {
        in = ((T*)data[n * 2 + i][ptroffset + blockIdx.y])[idx];

        if (PRINT && idx == 0 && blockIdx.y == 0)
            printf("LT:, b: %d, g:, in pt: %lu \n", i, (unsigned long)in);

        T aux0 = modmult<algo>(in, ((T*)data[i][ptroffset + blockIdx.y])[idx], primeid);

        if (PRINT && idx == 0 && blockIdx.y == 0)
            printf("LT: %d, b: %d, in c0: %lu \n", -1, i, (unsigned long)((T*)data[i][ptroffset + blockIdx.y])[idx]);

        T aux1 = modmult<algo>(in, ((T*)data[n + i][ptroffset + blockIdx.y])[idx], primeid);

        if (PRINT && idx == 0 && blockIdx.y == 0)
            printf("LT: %d, b: %d, in c1: %lu \n", -1, i,
                   (unsigned long)((T*)data[n + i][ptroffset + blockIdx.y])[idx]);
        out0 = modadd(out0, aux0, primeid);
        out1 = modadd(out1, aux1, primeid);
    }
    if (PRINT && idx == 0 && blockIdx.y == 0)
        printf("LT: , g: , res: %lu \n", (unsigned long)out0);
    if (PRINT && idx == 0 && blockIdx.y == 0)
        printf("LT: , g: , res: %lu \n", (unsigned long)out1);
    ((T*)c0[ptroffset + blockIdx.y])[idx] = out0;
    ((T*)c1[ptroffset + blockIdx.y])[idx] = out1;
}

__global__ void dotProductPt_(void** c0, void** c1, void*** data, const size_t ptroffset, const int primeidInit,
                              const int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = C_.primeid_flattened[primeidInit + blockIdx.y];

    if (ISU64(primeid))
        dotProductPtBody<uint64_t>(c0, c1, data, ptroffset, n, idx, primeid);
    else
        dotProductPtBody<uint32_t>(c0, c1, data, ptroffset, n, idx, primeid);
}

/*
__global__ void dotProductLtBatchedPt___(void*** c0_out, void*** c1_out, void*** c0_in, void*** c1_in, void*** pts,
                                         const int batch, const int gStep, const int primeidInit, const int n) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int& b = blockDim.z;
    const int primeid = C_.primeid_flattened[primeidInit + blockIdx.y];
    constexpr ALGO algo = ALGO_BARRETT;

    extern __shared__ char buffer[];

    // Shared required: (2*batch+1/2)*threads_per_block
    const int in_stride = blockDim.x * blockDim.y * blockDim.z;
    const int block_id = (threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z);
    uint64_t* in = ((uint64_t*)buffer) + block_id;

    __uint128_t* acc_this_thread = (__uint128_t*)(((uint64_t*)buffer) + in_stride * batch) + block_id;
    uint64_t* pt = ((uint64_t*)buffer) + (batch + 2) * in_stride + (threadIdx.x + blockDim.x * threadIdx.z);

    const bool im_c0 = threadIdx.y == 0;
    const int b_idx = threadIdx.z;

    if (PRINT && idx == 0 && blockIdx.y == 0 && im_c0 && b_idx == 0 && blockIdx.z == 0) {
        printf("in: %lu, acc: %lu, pt: %lu\n", in, acc_this_thread, pt);
    }

    void*** inputs = im_c0 ? c0_in : c1_in;
    void*** outputs = im_c0 ? c0_out : c1_out;

    for (int k = blockIdx.z; k < n; k += gridDim.z) {
        for (int i = 0; i < batch; ++i) {
            in[in_stride * i] = ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx];
            if (PRINT && idx == 0 && blockIdx.y == 0 && im_c0)
                printf("LT: %d, b: %d, in c0: %lu %lu %p %p\n", k, b_idx, in[in_stride * i],
                       ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx],
                       inputs[k * b * batch + i * b + b_idx], inputs);
            if (PRINT && idx == 0 && blockIdx.y == 0 && !im_c0)
                printf("LT: %d, b: %d, in c1: %lu %lu %p %p\n", k, b_idx, in[in_stride * i],
                       ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx],
                       inputs[k * b * batch + i * b + b_idx], inputs);
        }

        for (int j = 0; j < gStep; ++j) {
            void** pt_partition = pts[k * b * gStep + j * b + b_idx];
            if (pt_partition != nullptr) {
                if (!im_c0) {
                    pt[0] = ((uint64_t*)pt_partition[blockIdx.y])[idx];
                    if (PRINT && idx == 0 && blockIdx.y == 0)
                        printf("LT: %d, b: %d, g: %d, in pt: %lu %lu %p %p\n", k, b_idx, j, pt[0],
                               ((uint64_t*)pt_partition[blockIdx.y])[idx], pt_partition, pts);
                }
            }

            __syncwarp();
            for (int i = 0; i < batch; ++i) {
                if (pt_partition != nullptr) {
                    acc_this_thread[0] = (__uint128_t)in[in_stride * i] * pt[0];
                } else {
                    acc_this_thread[0] = 0;
                }

                if constexpr (0) {
                    __syncthreads();
                    if (threadIdx.z == 0) {
                        __uint128_t res = acc_this_thread[0];
                        for (int i = 1; i < blockDim.z; ++i) {
                            res = res + acc_this_thread[i * blockDim.x * blockDim.y];
                        }

                        ((uint64_t*)outputs[k * batch * gStep + i * gStep + j][blockIdx.y])[idx] =
                            modreduce<ALGO_NATIVE>(res, primeid);
                        if (PRINT && idx == 0 && blockIdx.y == 0)
                            printf("LT: %d, g: %d, res: %lu \n", k, j, acc_this_thread[0]);
                    }
                    __syncthreads();
                } else {
                    const int r_init = 1 << (32 - __clz(b - 1) - 1);
                    int r = r_init;
                    if (r > 0) {
                        __syncthreads();
                        if (threadIdx.z + r < b) {
                            acc_this_thread[0] = acc_this_thread[0] + acc_this_thread[0 + r * blockDim.x * blockDim.y];
                        }
                    }

                    r >>= 1;
                    for (; r > 0; r >>= 1) {
                        __syncthreads();
                        if (threadIdx.z < r) {
                            acc_this_thread[0] = acc_this_thread[0] + acc_this_thread[0 + r * blockDim.x * blockDim.y];
                        }
                    }
                    if (threadIdx.z == 0) {

                        ((uint64_t*)outputs[k * batch * gStep + i * gStep + j][blockIdx.y])[idx] =
                            modreduce<ALGO_NATIVE>(acc_this_thread[0], primeid);
                        if (PRINT && idx == 0 && blockIdx.y == 0)
                            printf("LT: %d, g: %d, res: %lu \n", k, j, acc_this_thread[0]);
                    }
                }
            }
        }
    }
}*/

__global__ void dotProductLtBatchedPt2___(void*** c0_out, void*** c1_out, void*** c0_in, void*** c1_in, void*** pts,
                                          const int bStep, const int gStep, const int primeidInit, const int n) {
    int idx = threadIdx.x + threadIdx.z * blockDim.x + blockIdx.x * blockDim.x * blockDim.z;
    //int b = blockDim.z;
    const int primeid = C_.primeid_flattened[primeidInit + blockIdx.y];
    constexpr ALGO algo = ALGO_BARRETT;

    extern __shared__ char buffer[];

    // Shared required: (2*batch+1/2)*threads_per_block
    const int in_stride = blockDim.x * blockDim.y * blockDim.z;
    const int block_id = (threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z);
    //uint64_t* in = ((uint64_t*)buffer) + block_id;
    //uint64_t in[6];

    //uint64_t* acc = ((uint64_t*)buffer) + in_stride * batch;
    uint64_t* acc_this_thread = ((uint64_t*)buffer) + block_id;
    //uint64_t* pt = acc + 2 * in_stride + (threadIdx.x + blockDim.x * threadIdx.z);

    //int r_init = 1 << (32 - __clz(b - 1) - 1);
    bool im_c0 = threadIdx.y == 0;
    const int b_idx = threadIdx.z;

    uint64_t pt;

    void*** inputs = im_c0 ? c0_in : c1_in;
    void*** outputs = im_c0 ? c0_out : c1_out;

    for (int k = blockIdx.z; k < n; k += gridDim.z) {

        for (int i = 0; i < bStep; ++i) {
            uint64_t in = ((uint64_t*)inputs[k * bStep + i][blockIdx.y])[idx];

            for (int j = 0; j < gStep; ++j) {
                void** pt_partition = pts[k * bStep * gStep + j * bStep + i];

                uint64_t mult = 0;
                if (pt_partition != nullptr) {
                    //pt[0] = ((uint64_t*)pt_partition[blockIdx.y])[idx];
                    mult = modmult<ALGO_BARRETT>(in, ((uint64_t*)pt_partition[blockIdx.y])[idx], primeid);
                }
                if (i == 0)
                    acc_this_thread[j * in_stride] = mult;
                else
                    acc_this_thread[j * in_stride] = modadd(acc_this_thread[j * in_stride], mult, primeid);
                if (i == bStep - 1)
                    ((uint64_t*)outputs[k * gStep + j][blockIdx.y])[idx] = acc_this_thread[j * in_stride];
            }
        }
    }
}

// n32: T = limb width, ACC = accumulator wide enough for bStep * (T*T).
// U64: 60-bit primes -> 120-bit products -> __uint128_t accumulator (as before).
// U32: 28-bit primes -> 56-bit products -> a uint64_t accumulator holds bStep up to 2^8=256
// terms without overflow, which covers every baby-step count this kernel is launched with.
// The host sizes the shared buffer with sizeof(__uint128_t) (LimbPartitionBatch.cu), so the
// U32 arm's uint64_t accumulator simply under-uses it — over-allocation is the safe direction.
template <typename T, typename ACC>
__device__ __forceinline__ void dotProductLtBatchedPt3Body(void*** c0_out, void*** c1_out, void*** c0_in,
                                                           void*** c1_in, void*** pts, const int bStep,
                                                           const int gStep, const int n, const int idx,
                                                           const int primeid, char* buffer) {
    const int in_stride = blockDim.x * blockDim.y * blockDim.z;
    const int block_id = (threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z);

    ACC* acc_this_thread = ((ACC*)buffer) + block_id;

    const bool im_c0 = threadIdx.y == 0;
    void*** inputs = im_c0 ? c0_in : c1_in;
    void*** outputs = im_c0 ? c0_out : c1_out;

    for (int k = blockIdx.z; k < n; k += gridDim.z) {
        for (int i = 0; i < bStep; ++i) {
            const T in = ((T*)inputs[k * bStep + i][blockIdx.y])[idx];

            for (int j = 0; j < gStep; ++j) {
                void** pt_partition = pts[k * bStep * gStep + j * bStep + i];

                ACC mult = 0;
                if (pt_partition != nullptr) {
                    mult = (ACC)in * (ACC)((T*)pt_partition[blockIdx.y])[idx];
                }
                if (i == 0)
                    acc_this_thread[j * in_stride] = mult;
                else
                    acc_this_thread[j * in_stride] = acc_this_thread[j * in_stride] + mult;

                if (i == bStep - 1) {
                    // modreduce(__uint128_t)->uint64_t and modreduce(uint64_t)->uint32_t both exist
                    const T res = modreduce<ALGO_NATIVE>(acc_this_thread[j * in_stride], primeid);
                    ((T*)outputs[k * gStep + j][blockIdx.y])[idx] = res;
                }
            }
        }
    }
}

// n32: this is THE live CoeffsToSlots / SlotsToCoeffs / linear-transform dot product
// (LinearTransform.cu -> RNSPoly::LTdotProductPtBatch -> LimbPartitionBatch.cu, VER2==VER3==true).
// It was 100% uint64_t-hardcoded with no width branch, so against U32 limbs every load fused
// coefficients 2*idx and 2*idx+1 into one word and every store splattered 8 bytes across two
// logical coefficients — the homomorphic DFT was decorrelated from its input. It did not fault
// only because U32 limbs are over-allocated to 2N elements (Limb.cu:34).
__global__ void dotProductLtBatchedPt3___(void*** c0_out, void*** c1_out, void*** c0_in, void*** c1_in, void*** pts,
                                          const int bStep, const int gStep, const int primeidInit, const int n) {
    int idx = threadIdx.x + threadIdx.z * blockDim.x + blockIdx.x * blockDim.x * blockDim.z;
    const int primeid = C_.primeid_flattened[primeidInit + blockIdx.y];

    extern __shared__ char buffer[];

    if (ISU64(primeid))
        dotProductLtBatchedPt3Body<uint64_t, __uint128_t>(c0_out, c1_out, c0_in, c1_in, pts, bStep, gStep, n, idx,
                                                          primeid, buffer);
    else
        dotProductLtBatchedPt3Body<uint32_t, uint64_t>(c0_out, c1_out, c0_in, c1_in, pts, bStep, gStep, n, idx,
                                                       primeid, buffer);
}

__global__ void dotProductLtBatchedPt___(void*** c0_out, void*** c1_out, void*** c0_in, void*** c1_in, void*** pts,
                                         const int batch, const int gStep, const int primeidInit, const int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int b = blockDim.z;
    const int primeid = C_.primeid_flattened[primeidInit + blockIdx.y];
    constexpr ALGO algo = ALGO_BARRETT;

    extern __shared__ char buffer[];

    // Shared required: (2*batch+1/2)*threads_per_block
    const int in_stride = blockDim.x * blockDim.y * blockDim.z;
    const int block_id = (threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z);
    uint64_t* in = ((uint64_t*)buffer) + block_id;
    //uint64_t in[6];

    uint64_t* acc = ((uint64_t*)buffer) + in_stride * batch;
    uint64_t* acc_this_thread = acc + block_id;
    uint64_t* pt = acc + 2 * in_stride + (threadIdx.x + blockDim.x * threadIdx.z);

    int r_init = 1 << (32 - __clz(b - 1) - 1);
    bool im_c0 = threadIdx.y == 0;
    const int b_idx = threadIdx.z;

    if (PRINT && idx == 0 && blockIdx.y == 0 && im_c0 && b_idx == 0 && blockIdx.z == 0) {
        printf("in: %lu, acc: %lu, pt: %lu\n", in, acc_this_thread, pt);
    }

    if (PRINT && idx == 0 && blockIdx.y == 0 && im_c0 && b_idx == 0 && blockIdx.z == 0)
        printf("%d <- 2^{floor(log2(bStep))}\n", r_init);

    void*** inputs = im_c0 ? c0_in : c1_in;
    void*** outputs = im_c0 ? c0_out : c1_out;

    for (int k = blockIdx.z; k < n; k += gridDim.z) {
        for (int i = 0; i < batch; ++i) {
            in[in_stride * i] = ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx];
            if (PRINT && idx == 0 && blockIdx.y == 0 && im_c0)
                printf("LT: %d, b: %d, in c0: %lu %lu %p %p\n", k, b_idx, in[/*in_stride * */ i],
                       ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx],
                       inputs[k * b * batch + i * b + b_idx], inputs);
            if (PRINT && idx == 0 && blockIdx.y == 0 && !im_c0)
                printf("LT: %d, b: %d, in c1: %lu %lu %p %p\n", k, b_idx, in[/*in_stride * */ i],
                       ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx],
                       inputs[k * b * batch + i * b + b_idx], inputs);
        }

        for (int j = 0; j < gStep; ++j) {
            void** pt_partition = pts[k * b * gStep + j * b + b_idx];
            if (pt_partition != nullptr) {
                if (im_c0) {
                    pt[0] = ((uint64_t*)pt_partition[blockIdx.y])[idx];
                    if (PRINT && idx == 0 && blockIdx.y == 0)
                        printf("LT: %d, b: %d, g: %d, in pt: %lu %lu %p %p\n", k, b_idx, j, pt[0],
                               ((uint64_t*)pt_partition[blockIdx.y])[idx], pt_partition, pts);
                }
            }

            __syncwarp();
            for (int i_0 = 0; i_0 < batch; i_0 += 2) {

                if (pt_partition != nullptr) {
                    acc_this_thread[0] = modmult<ALGO_BARRETT>(in[in_stride * i_0], pt[0], primeid);
                    if (i_0 + 1 < batch)
                        acc_this_thread[in_stride] = modmult<ALGO_BARRETT>(in[in_stride * (i_0 + 1)], pt[0], primeid);
                } else {
                    acc_this_thread[0] = 0;
                    acc_this_thread[in_stride] = 0;
                }

                {
                    int r = r_init;
                    if (r > 0) {
                        __syncthreads();
                        if (threadIdx.z + r < b) {
                            acc_this_thread[0] =
                                modadd(acc_this_thread[0], acc_this_thread[0 + r * blockDim.x * blockDim.y], primeid);
                        }
                        if (threadIdx.z >= r) {
                            acc_this_thread[in_stride] =
                                modadd(acc_this_thread[in_stride],
                                       acc_this_thread[in_stride - r * blockDim.x * blockDim.y], primeid);
                        }
                    }

                    r >>= 1;
                    for (; r > 0; r >>= 1) {
                        __syncthreads();
                        if (threadIdx.z < r) {
                            acc_this_thread[0] =
                                modadd(acc_this_thread[0], acc_this_thread[0 + r * blockDim.x * blockDim.y], primeid);
                        }
                        if (threadIdx.z >= b - r) {
                            acc_this_thread[in_stride] =
                                modadd(acc_this_thread[in_stride],
                                       acc_this_thread[in_stride - r * blockDim.x * blockDim.y], primeid);
                        }
                    }
                    if (threadIdx.z == 0) {
                        ((uint64_t*)outputs[k * batch * gStep + i_0 * gStep + j][blockIdx.y])[idx] = acc_this_thread[0];
                        if (PRINT && idx == 0 && blockIdx.y == 0)
                            printf("LT: %d, g: %d, res: %lu \n", k, j, acc_this_thread[0]);
                    }
                    if (threadIdx.z == b - 1) {
                        if (i_0 + 1 < batch)
                            ((uint64_t*)outputs[k * batch * gStep + (i_0 + 1) * gStep + j][blockIdx.y])[idx] =
                                acc_this_thread[in_stride];
                    }
                }
            }
        }
    }
}

/*

__global__ void dotProductLtBatchedPt___(void*** c0_out, void*** c1_out, void*** c0_in, void*** c1_in, void*** pts,
                                         const int batch, const int gStep, const int primeidInit, const int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int b = blockDim.z;
    const int primeid = C_.primeid_flattened[primeidInit + blockIdx.y];
    constexpr ALGO algo = ALGO_BARRETT;

    extern __shared__ char buffer[];

    // Shared required: (2*batch+1/2)*threads_per_block
    const int in_stride = blockDim.x * blockDim.y * blockDim.z;
    const int block_id = (threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z);
    uint64_t* in = ((uint64_t*)buffer) + block_id;

    uint64_t* acc = ((uint64_t*)buffer) + in_stride * batch;
    uint64_t* acc_this_thread = acc + block_id;
    uint64_t* pt = acc + in_stride * batch + (threadIdx.x + blockDim.x * threadIdx.z);

    int r_init = 1 << (32 - __clz(b - 1) - 1);
    bool im_c0 = threadIdx.y == 0;
    const int b_idx = threadIdx.z;

    if (PRINT && idx == 0 && blockIdx.y == 0 && im_c0 && b_idx == 0 && blockIdx.z == 0) {
        printf("in: %lu, acc: %lu, pt: %lu\n", in, acc_this_thread, pt);
    }

    if (PRINT && idx == 0 && blockIdx.y == 0 && im_c0 && b_idx == 0 && blockIdx.z == 0)
        printf("%d <- 2^{floor(log2(bStep))}\n", r_init);

    void*** inputs = im_c0 ? c0_in : c1_in;
    void*** outputs = im_c0 ? c0_out : c1_out;

    for (int k = blockIdx.z; k < n; k += gridDim.z) {
        for (int i = 0; i < batch; ++i) {
            in[in_stride * i] = ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx];
            if (PRINT && idx == 0 && blockIdx.y == 0 && im_c0)
                printf("LT: %d, b: %d, in c0: %lu %lu %p %p\n", k, b_idx, in[in_stride * i],
                       ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx],
                       inputs[k * b * batch + i * b + b_idx], inputs);
            if (PRINT && idx == 0 && blockIdx.y == 0 && !im_c0)
                printf("LT: %d, b: %d, in c1: %lu %lu %p %p\n", k, b_idx, in[in_stride * i],
                       ((uint64_t*)inputs[k * b * batch + i * b + b_idx][blockIdx.y])[idx],
                       inputs[k * b * batch + i * b + b_idx], inputs);
        }

        for (int j = 0; j < gStep; ++j) {
            void** pt_partition = pts[k * b * gStep + j * b + b_idx];
            if (pt_partition != nullptr) {
                if (im_c0) {
                    pt[0] = ((uint64_t*)pt_partition[blockIdx.y])[idx];
                    if (PRINT && idx == 0 && blockIdx.y == 0)
                        printf("LT: %d, b: %d, g: %d, in pt: %lu %lu %p %p\n", k, b_idx, j, pt[0],
                               ((uint64_t*)pt_partition[blockIdx.y])[idx], pt_partition, pts);
                }
            }

            __syncwarp();
            for (int i = 0; i < batch; ++i) {
                if (pt_partition != nullptr) {
                    acc_this_thread[in_stride * i] = modmult<ALGO_BARRETT>(in[in_stride * i], pt[0], primeid);
                } else {
                    acc_this_thread[in_stride * i] = 0;
                }
            }

            int r = r_init;
            if (r > 0) {
                __syncthreads();
                if (threadIdx.z + r < b) {
                    for (int i = 0; i < batch; ++i) {
                        acc_this_thread[in_stride * i] =
                            modadd(acc_this_thread[in_stride * i],
                                   acc_this_thread[in_stride * i + r * blockDim.x * blockDim.y], primeid);
                    }
                }
            }

            r >>= 1;
            for (; r > 0; r >>= 1) {
                __syncthreads();
                if (threadIdx.z < r) {
                    for (int i = 0; i < batch; ++i) {
                        acc_this_thread[in_stride * i] =
                            modadd(acc_this_thread[in_stride * i],
                                   acc_this_thread[in_stride * i + r * blockDim.x * blockDim.y], primeid);
                    }
                }
            }
            if (threadIdx.z == 0) {

                for (int i = 0; i < batch; ++i) {
                    ((uint64_t*)outputs[k * batch * gStep + i * gStep + j][blockIdx.y])[idx] =
                        acc_this_thread[in_stride * i];
                    if (PRINT && idx == 0 && blockIdx.y == 0)
                        printf("LT: %d, g: %d, res: %lu \n", k, j, acc_this_thread[in_stride * i]);
                }
            }
        }
    }
}

 */
__global__ void addScaleB_(void** a, void** b, void** c, const int primeid_init) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        uint64_t in = modmult<ALGO_SHOUP>(((uint64_t*)b[blockIdx.y])[idx], C_.P[primeid], primeid, C_.P_shoup[primeid]);
        ((uint64_t*)a[blockIdx.y])[idx] = modadd(in, ((uint64_t*)c[blockIdx.y])[idx], primeid);
    } else {
        uint32_t in = modmult<ALGO_SHOUP>(((uint32_t*)b[blockIdx.y])[idx], (uint32_t)C_.P[primeid], primeid,
                                          (uint32_t)C_.P_shoup[primeid]);
        ((uint32_t*)a[blockIdx.y])[idx] = modadd(in, ((uint32_t*)c[blockIdx.y])[idx], primeid);
    }
}

__global__ void scaleByP_(void** a, const int primeid_init) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        uint64_t in = modmult<ALGO_SHOUP>(((uint64_t*)a[blockIdx.y])[idx], C_.P[primeid], primeid, C_.P_shoup[primeid]);
        ((uint64_t*)a[blockIdx.y])[idx] = in;
    } else {
        uint32_t in = modmult<ALGO_SHOUP>(((uint32_t*)a[blockIdx.y])[idx], (uint32_t)C_.P[primeid], primeid,
                                          (uint32_t)C_.P_shoup[primeid]);
        ((uint32_t*)a[blockIdx.y])[idx] = in;
    }
}

__global__ void add_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modadd(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
            }
        }
    } else {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modadd(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
            }
        }
    }
}

__global__ void add_reuse_scale_p_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
            load = modmult<ALGO_SHOUP>(load, C_.P[primeid], primeid, C_.P_shoup[primeid]);
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modadd(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
            }
        }
    } else {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
            load = modmult<ALGO_SHOUP>(load, (uint32_t)C_.P[primeid], primeid, (uint32_t)C_.P_shoup[primeid]);
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modadd(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
            }
        }
    }
}

__global__ void sub_reuse_scale_p_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
            load = modmult<ALGO_SHOUP>(load, C_.P[primeid], primeid, C_.P_shoup[primeid]);
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modsub(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
            }
        }
    } else {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
            load = modmult<ALGO_SHOUP>(load, (uint32_t)C_.P[primeid], primeid, (uint32_t)C_.P_shoup[primeid]);
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modsub(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
            }
        }
    }
}

__global__ void add_scale_p_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                uint64_t aux = modmult<ALGO_SHOUP>(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], C_.P[primeid],
                                                   primeid, C_.P_shoup[primeid]);
                ((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] = modadd(aux, load, primeid);
            }
        }
    } else {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                uint32_t aux = modmult<ALGO_SHOUP>(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx],
                                                   (uint32_t)C_.P[primeid], primeid, (uint32_t)C_.P_shoup[primeid]);
                ((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] = modadd(aux, load, primeid);
            }
        }
    }
}

__global__ void sub_scale_p_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                uint64_t aux = modmult<ALGO_SHOUP>(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], C_.P[primeid],
                                                   primeid, C_.P_shoup[primeid]);
                ((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] = modsub(aux, load, primeid);
            }
        }
    } else {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                uint32_t aux = modmult<ALGO_SHOUP>(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx],
                                                   (uint32_t)C_.P[primeid], primeid, (uint32_t)C_.P_shoup[primeid]);
                ((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] = modsub(aux, load, primeid);
            }
        }
    }
}

__global__ void copy_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] = load;
            }
        }
    } else {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] = load;
            }
        }
    }
}

__global__ void copy_reuse_negative_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
            load = load == 0 ? 0 : C_.primes[primeid] - load;
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] = load;
            }
        }
    } else {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
            load = load == 0 ? 0 : C_.primes[primeid] - load;
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] = load;
            }
        }
    }
}

__global__ void add_scalar_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint64_t load = ((uint64_t*)b)[i * MAXP + primeid];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modadd(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
            }
        }
    } else {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint32_t load = ((uint32_t*)b)[i * MAXP + primeid];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modadd(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
            }
        }
    }
}

__global__ void mult_scalar_reuse_b___(void*** a, void*** b, void*** b_shoup, const int primeid_init, const int n,
                                       const int its) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint64_t load = ((uint64_t*)b)[i * MAXP + primeid];
            uint64_t load_shoup = ((uint64_t*)b_shoup)[i * MAXP + primeid];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modmult<ALGO_SHOUP>(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid, load_shoup);
            }
        }
    } else {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint32_t load = ((uint32_t*)b)[i * MAXP + primeid];
            uint32_t load_shoup = ((uint32_t*)b_shoup)[i * MAXP + primeid];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modmult<ALGO_SHOUP>(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid, load_shoup);
            }
        }
    }
}

__global__ void sub_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modsub(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
            }
        }
    } else {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modsub(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
            }
        }
    }
}

__global__ void mult_reuse_b___(void*** a, void*** b, const int primeid_init, const int n, const int its) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint64_t load = ((uint64_t*)(b[i][blockIdx.y]))[idx];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint64_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modmult<ALGO_BARRETT>(((uint64_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
            }
        }
    } else {
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            uint32_t load = ((uint32_t*)(b[i][blockIdx.y]))[idx];
            for (int j = 0; j < its && a[i * its + j] != nullptr; ++j) {
                ((uint32_t*)(a[i * its + j][blockIdx.y]))[idx] =
                    modmult<ALGO_BARRETT>(((uint32_t*)(a[i * its + j][blockIdx.y]))[idx], load, primeid);
            }
        }
    }
}

__global__ void binomialMult_(const __grid_constant__ int primeid_init, void** c0, void** c1, void** c2, void** d0,
                              void** d1) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr ALGO algo = ALGO_BARRETT;

    if (ISU64(primeid)) {
        using T = uint64_t;
        T d0in = ((T*)(d0[blockIdx.y]))[idx];
        T d1in = ((T*)(d1[blockIdx.y]))[idx];
        T c0in = ((T*)(c0[blockIdx.y]))[idx];
        T c1in = ((T*)(c1[blockIdx.y]))[idx];

        T aux0 = modmult<ALGO_BARRETT>(c0in, d0in, primeid);
        ((T*)(c0[blockIdx.y]))[idx] = aux0;

        T aux1 =
            modadd(modmult<ALGO_BARRETT>(c0in, d1in, primeid), modmult<ALGO_BARRETT>(c1in, d0in, primeid), primeid);
        ((T*)(c1[blockIdx.y]))[idx] = aux1;

        T aux2 = modmult<ALGO_BARRETT>(c1in, d1in, primeid);
        ((T*)(c2[blockIdx.y]))[idx] = aux2;

    } else {
        using T = uint32_t;
        T d0in = ((T*)(d0[blockIdx.y]))[idx];
        T d1in = ((T*)(d1[blockIdx.y]))[idx];
        T c0in = ((T*)(c0[blockIdx.y]))[idx];
        T c1in = ((T*)(c1[blockIdx.y]))[idx];

        T aux0 = modmult<ALGO_BARRETT>(c0in, d0in, primeid);
        ((T*)(c0[blockIdx.y]))[idx] = aux0;

        T aux1 =
            modadd(modmult<ALGO_BARRETT>(c0in, d1in, primeid), modmult<ALGO_BARRETT>(c1in, d0in, primeid), primeid);
        ((T*)(c1[blockIdx.y]))[idx] = aux1;

        T aux2 = modmult<ALGO_BARRETT>(c1in, d1in, primeid);
        ((T*)(c2[blockIdx.y]))[idx] = aux2;
    }
}

__global__ void binomialMultExtend_(const __grid_constant__ int primeid_init, void** c0, void** c1, void** c2,
                                    void** d0, void** d1) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr ALGO algo = ALGO_BARRETT;

    if (ISU64(primeid)) {
        using T = uint64_t;
        T d0in = ((T*)(d0[blockIdx.y]))[idx];
        T d1in = ((T*)(d1[blockIdx.y]))[idx];
        T c0in = ((T*)(c0[blockIdx.y]))[idx];
        T c1in = ((T*)(c1[blockIdx.y]))[idx];

        T aux0 = modmult<ALGO_BARRETT>(c0in, d0in, primeid);
        ((T*)(c0[blockIdx.y]))[idx] = modmult<ALGO_SHOUP>(aux0, (T)C_.P[primeid], primeid, (T)C_.P_shoup[primeid]);

        T aux1 =
            modadd(modmult<ALGO_BARRETT>(c0in, d1in, primeid), modmult<ALGO_BARRETT>(c1in, d0in, primeid), primeid);
        ((T*)(c1[blockIdx.y]))[idx] = modmult<ALGO_SHOUP>(aux1, (T)C_.P[primeid], primeid, (T)C_.P_shoup[primeid]);

        T aux2 = modmult<ALGO_BARRETT>(c1in, d1in, primeid);
        ((T*)(c2[blockIdx.y]))[idx] = aux2;

    } else {
        using T = uint32_t;
        T d0in = ((T*)(d0[blockIdx.y]))[idx];
        T d1in = ((T*)(d1[blockIdx.y]))[idx];
        T c0in = ((T*)(c0[blockIdx.y]))[idx];
        T c1in = ((T*)(c1[blockIdx.y]))[idx];

        T aux0 = modmult<ALGO_BARRETT>(c0in, d0in, primeid);
        ((T*)(c0[blockIdx.y]))[idx] = modmult<ALGO_SHOUP>(aux0, (T)C_.P[primeid], primeid, (T)C_.P_shoup[primeid]);

        T aux1 =
            modadd(modmult<ALGO_BARRETT>(c0in, d1in, primeid), modmult<ALGO_BARRETT>(c1in, d0in, primeid), primeid);
        ((T*)(c1[blockIdx.y]))[idx] = modmult<ALGO_SHOUP>(aux1, (T)C_.P[primeid], primeid, (T)C_.P_shoup[primeid]);

        T aux2 = modmult<ALGO_BARRETT>(c1in, d1in, primeid);
        ((T*)(c2[blockIdx.y]))[idx] = aux2;
    }
}

__global__ void binomialSquare_(const __grid_constant__ int primeid_init, void** c0, void** c1, void** c2) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr ALGO algo = ALGO_BARRETT;

    if (ISU64(primeid)) {
        using T = uint64_t;
        T c0in = ((T*)(c0[blockIdx.y]))[idx];
        T c1in = ((T*)(c1[blockIdx.y]))[idx];

        T aux0 = modmult<ALGO_BARRETT>(c0in, c0in, primeid);
        ((T*)(c0[blockIdx.y]))[idx] = aux0;

        T aux1 = modmult<ALGO_BARRETT>(c0in, c1in, primeid);
        ((T*)(c1[blockIdx.y]))[idx] = modadd(aux1, aux1, primeid);

        T aux2 = modmult<ALGO_BARRETT>(c1in, c1in, primeid);
        ((T*)(c2[blockIdx.y]))[idx] = aux2;

    } else {
        using T = uint32_t;
        T c0in = ((T*)(c0[blockIdx.y]))[idx];
        T c1in = ((T*)(c1[blockIdx.y]))[idx];

        T aux0 = modmult<ALGO_BARRETT>(c0in, c0in, primeid);
        ((T*)(c0[blockIdx.y]))[idx] = aux0;

        T aux1 = modmult<ALGO_BARRETT>(c0in, c1in, primeid);
        ((T*)(c1[blockIdx.y]))[idx] = modadd(aux1, aux1, primeid);

        T aux2 = modmult<ALGO_BARRETT>(c1in, c1in, primeid);
        ((T*)(c2[blockIdx.y]))[idx] = aux2;
    }
}

__global__ void binomialSquareExtend_(const __grid_constant__ int primeid_init, void** c0, void** c1, void** c2) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr ALGO algo = ALGO_BARRETT;

    if (ISU64(primeid)) {
        using T = uint64_t;
        T c0in = ((T*)(c0[blockIdx.y]))[idx];
        T c1in = ((T*)(c1[blockIdx.y]))[idx];

        T aux0 = modmult<ALGO_BARRETT>(c0in, c0in, primeid);
        ((T*)(c0[blockIdx.y]))[idx] = modmult<ALGO_SHOUP>(aux0, C_.P[primeid], primeid, C_.P_shoup[primeid]);

        T aux1 = modmult<ALGO_BARRETT>(c0in, c1in, primeid);
        ((T*)(c1[blockIdx.y]))[idx] =
            modmult<ALGO_SHOUP>(modadd(aux1, aux1, primeid), C_.P[primeid], primeid, C_.P_shoup[primeid]);

        T aux2 = modmult<ALGO_BARRETT>(c1in, c1in, primeid);
        ((T*)(c2[blockIdx.y]))[idx] = aux2;

    } else {
        using T = uint32_t;
        T c0in = ((T*)(c0[blockIdx.y]))[idx];
        T c1in = ((T*)(c1[blockIdx.y]))[idx];

        T aux0 = modmult<ALGO_BARRETT>(c0in, c0in, primeid);
        ((T*)(c0[blockIdx.y]))[idx] = modmult<ALGO_SHOUP>(aux0, (T)C_.P[primeid], primeid, (T)C_.P_shoup[primeid]);

        T aux1 = modmult<ALGO_BARRETT>(c0in, c1in, primeid);
        ((T*)(c1[blockIdx.y]))[idx] =
            modmult<ALGO_SHOUP>(modadd(aux1, aux1, primeid), (T)C_.P[primeid], primeid, (T)C_.P_shoup[primeid]);

        T aux2 = modmult<ALGO_BARRETT>(c1in, c1in, primeid);
        ((T*)(c2[blockIdx.y]))[idx] = aux2;
    }
}

__global__ void binomialDotProdBatched___(const __grid_constant__ int primeid_init, void*** c0, void*** c1, void*** d0,
                                          void*** d1, void*** c0_out, void*** c1_out, void*** c2_out, int its, int n) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    constexpr ALGO algo = ALGO_BARRETT;

    if (ISU64(primeid)) {
        using T = uint64_t;
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            T d0in = ((T*)(d0[i][blockIdx.y]))[idx];
            T d1in = ((T*)(d1[i][blockIdx.y]))[idx];
            T acc0, acc1, acc2;
            for (int j = 0; j < its && c0[i * its + j] != nullptr; ++j) {
                T c0in = ((T*)(c0[i * its + j][blockIdx.y]))[idx];
                T c1in = ((T*)(c1[i * its + j][blockIdx.y]))[idx];

                T aux0 = modmult<ALGO_BARRETT>(c0in, d0in, primeid);

                T aux1 = modadd(modmult<ALGO_BARRETT>(c0in, d1in, primeid), modmult<ALGO_BARRETT>(c1in, d0in, primeid),
                                primeid);

                T aux2 = modmult<ALGO_BARRETT>(c1in, d1in, primeid);

                if (j == 0) {
                    acc0 = aux0;
                    acc1 = aux1;
                    acc2 = aux2;
                } else {
                    acc0 = modadd(acc0, aux0, primeid);
                    acc1 = modadd(acc1, aux1, primeid);
                    acc2 = modadd(acc2, aux2, primeid);
                }
            }

            ((T*)(c0_out[i][blockIdx.y]))[idx] = acc0;
            ((T*)(c1_out[i][blockIdx.y]))[idx] = acc1;
            ((T*)(c2_out[i][blockIdx.y]))[idx] = acc2;
        }
    } else {
        using T = uint32_t;
        for (int i = blockIdx.z; i < n / its; i += gridDim.z) {
            T d0in = ((T*)(d0[i][blockIdx.y]))[idx];
            T d1in = ((T*)(d1[i][blockIdx.y]))[idx];
            T acc0, acc1, acc2;
            for (int j = 0; j < its && c0[i * its + j] != nullptr; ++j) {
                T c0in = ((T*)(c0[i * its + j][blockIdx.y]))[idx];
                T c1in = ((T*)(c1[i * its + j][blockIdx.y]))[idx];

                T aux0 = modmult<ALGO_BARRETT>(c0in, d0in, primeid);

                T aux1 = modadd(modmult<ALGO_BARRETT>(c0in, d1in, primeid), modmult<ALGO_BARRETT>(c1in, d0in, primeid),
                                primeid);

                T aux2 = modmult<ALGO_BARRETT>(c1in, d1in, primeid);

                if (j == 0) {
                    acc0 = aux0;
                    acc1 = aux1;
                    acc2 = aux2;
                } else {
                    acc0 = modadd(acc0, aux0, primeid);
                    acc1 = modadd(acc1, aux1, primeid);
                    acc2 = modadd(acc2, aux2, primeid);
                }
            }

            ((T*)(c0_out[i][blockIdx.y]))[idx] = acc0;
            ((T*)(c1_out[i][blockIdx.y]))[idx] = acc1;
            ((T*)(c2_out[i][blockIdx.y]))[idx] = acc2;
        }
    }
}

}  // namespace CKKS
}  // namespace FIDESlib

#define YY(algo)                                                 \
    template __global__ void FIDESlib::CKKS::Scalar_mult_<algo>( \
        void** a, const uint64_t* b, const __grid_constant__ int primeid_init, const uint64_t* shoup_mu);
#include "ntt_types.inc"
#undef YY

template __global__ void FIDESlib::CKKS::addMult_<uint64_t>(uint64_t* l, const uint64_t* l1, const uint64_t* l2,
                                                            const __grid_constant__ int primeid);

template __global__ void FIDESlib::CKKS::addMult_<uint32_t>(uint32_t* l, const uint32_t* l1, const uint32_t* l2,
                                                            const __grid_constant__ int primeid);
