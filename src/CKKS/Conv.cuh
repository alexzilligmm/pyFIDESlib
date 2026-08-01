//
// Created by carlosad on 24/03/24.
//

#ifndef FIDESLIB_CKKS_CONV_CUH
#define FIDESLIB_CKKS_CONV_CUH

#include "ModMult.cuh"

namespace FIDESlib::CKKS {
template <typename T>
__global__ void conv1_(T* a, const T q_hat_inv, const int primeid);

/**
 * Dynamic shared memory should be sizeof(T) * (K + blockDim.y) * blockDim.y
 * @tparam T
 * @param a
 * @param typea
 * @param n
 * @param b
 * @param G_
 * @param typeb
 * @param m
 * @param L
 */
template <ALGO algo = ALGO_SHOUP>
__global__ void ModDown2(void** __restrict__ a, const __grid_constant__ int n, void** __restrict__ b,
                         const __grid_constant__ int primeid_init, const Global::Globals* Globals);

/**
 * Level/window bookkeeping for ONE DecompAndModUpConv launch, all host-computed.
 *
 * On a classic prefix chain every field is derivable from (n, d) through the
 * num_primeid_digit_* / level tables, and a DEFAULT-CONSTRUCTED instance makes the kernel
 * derive exactly what it derived before — the classic launch sites need no edit and the
 * generated arms are unchanged.
 *
 * On an RR chain (RR_PLAN milestone (c).3) none of it is derivable: a digit is
 * window ∩ global partition, so it is truncated at BOTH ends, and the level index is the
 * rescale count rather than the limb count. The digit's active source primes are still a
 * contiguous run of the digit's global list (offset `fromOff`), and its active destinations
 * are still the specials followed by ONE contiguous run of the digit's global destination
 * list — the run starts at window-lo because the removed digit block sits in the middle of
 * [lo, hi], which is what `toOff` shifts by.
 */
struct ModUpWindow {
    int fromOff = 0;    //!< start offset into the digit's DECOMP list (slot AND prime)
    int toOff = 0;      //!< slot shift of the Q part of the digit's DIGIT list (0 = prefix)
    int nSpecial = 0;   //!< specials at the head of the DIGIT list; only consulted if toOff != 0
    int nFrom = -1;     //!< active source count      (-1 = C_.num_primeid_digit_from[d][n-1])
    int nTo = -1;       //!< active destination count (-1 = C_.num_primeid_digit_to[d][n-1])
    int scaleIdx = -1;  //!< middle index of DecompAndModUp_pre_scale (-1 = nFrom-1, i.e. digit size)
    int matIdx = -1;    //!< level index of DecompAndModUp_matrix      (-1 = n-1)
    int winLo = 0;      //!< active-window low  (destination guard)
    int winHi = -1;     //!< active-window high (destination guard; -1 = n-1)
};

template <ALGO algo = ALGO_SHOUP>
__global__ void DecompAndModUpConv(void** __restrict__ a, const int __grid_constant__ n, void** __restrict__ b,
                                   const int __grid_constant__ d, const Global::Globals* Globals,
                                   const ModUpWindow w = ModUpWindow{});
}  // namespace FIDESlib::CKKS
#endif  //FIDESLIB_CKKS_CONV_CUH
