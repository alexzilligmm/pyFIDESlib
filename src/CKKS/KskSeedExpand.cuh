//
// Lever 1b-ii (KSK seed expansion): deterministic, random-access expansion of the
// key-switching-key `a` component from a 256-bit per-key seed.
//
// ============================== FROZEN SPEC v1 ==============================
// Generator: ChaCha12 in counter mode (USER RULING 2026-07-27: crypto-grade expander,
// NOT Philox — seed-expanded public `a` is standard practice only with a cryptographic
// XOF; ChaCha is GPU-native. 12 rounds; the block function below takes the round count
// so ChaCha20 is a one-line swap at the call sites).
//
// State layout (standard ChaCha constants row):
//   s[0..3]  = "expa" "nd 3" "2-by" "te k"  (0x61707865, 0x3320646e, 0x79622d32, 0x6b206574)
//   s[4..11] = key[0..7]           256-bit per-KSK seed
//   s[12]    = block counter       (see addressing below)
//   s[13]    = digit index j       (hybrid-keyswitch digit, 0..dnum-1)
//   s[14]    = modulus p           the limb's RNS modulus VALUE — self-describing limb tag,
//                                  enumeration-order-independent between CPU and GPU
//   s[15]    = 0x4b534b41          "KSKA" domain separator
// 12 rounds (6 double-rounds) + feedforward add. Little-endian u32 words throughout
// (both platforms are little-endian; no byte-serialization is ever performed).
//
// Coefficient addressing (per (key, digit j, modulus p) stream, ring dim N, slot s < N):
//   base block  b0  = s >> 4      (one 16-word block serves 16 consecutive slots)
//   word        w   = s & 15
//   escalation stride ESC = N >> 4
//   attempt t = 0..T_MAX-1 reads word w of block (b0 + t*ESC):
//     m = 0xFFFFFFFF / p  (== floor(2^32/p) since p is an odd prime)
//     accept the first v with v < m*p  ->  coefficient = v % p   (exact-uniform rejection)
//   all T_MAX rejected (prob < (p/2^32)^16 < 2^-64 per coeff): coefficient = last v % p.
// T_MAX = 16. Rejection prob per attempt < p/2^32 < 2^-4 for this campaign's p < 2^28,
// so the expected extra-block rate is < 2^-4 per coefficient — the stage-3 kernel path
// amortizes one block per 16 slots and takes the rare escalation divergently.
//
// `a` is sampled by OpenFHE DIRECTLY in EVALUATION format (poly-impl.h:64,
// keyswitch-hybrid.cpp:103 — verified 2026-07-27), so slot s is literally the eval-format
// coefficient index on both platforms; loadDecompDigit copies host vectors verbatim.
//
// THE OPENFHE PATCH CARRIES A PLAIN-C++ COPY OF THIS FILE (the two trees cannot share a
// header); the test_ksk_seed_expand parity gate is the guard that the copies stay
// bit-identical. Any change here is a SPEC change: bump the domain separator and re-gate.
// ============================================================================
//

#ifndef FIDESLIB_CKKS_KSKSEEDEXPAND_CUH
#define FIDESLIB_CKKS_KSKSEEDEXPAND_CUH

#include <cstdint>

#if defined(__CUDACC__)
#define FIDESLIB_KSK_HD __host__ __device__ __forceinline__
#else
#define FIDESLIB_KSK_HD inline
#endif

namespace FIDESlib::CKKS::kskexpand {

constexpr uint32_t kDomainSep = 0x4b534b41u;  // "KSKA"
constexpr int kTMax = 16;
constexpr int kRounds = 12;  // ChaCha12 (the shippable crypto margin choice; 20 = one-line swap)

FIDESLIB_KSK_HD uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

FIDESLIB_KSK_HD void quarterround(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
    a += b; d ^= a; d = rotl32(d, 16);
    c += d; b ^= c; b = rotl32(b, 12);
    a += b; d ^= a; d = rotl32(d, 8);
    c += d; b ^= c; b = rotl32(b, 7);
}

/* One ChaCha block (rounds = kRounds) for the KSKA domain: out[16] = keystream words. */
FIDESLIB_KSK_HD void chacha_block(const uint32_t key[8], uint32_t block_ctr, uint32_t digit, uint32_t modulus,
                                  uint32_t out[16]) {
    uint32_t s[16] = {0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u,
                      key[0],      key[1],      key[2],      key[3],
                      key[4],      key[5],      key[6],      key[7],
                      block_ctr,   digit,       modulus,     kDomainSep};
    uint32_t x0 = s[0], x1 = s[1], x2 = s[2], x3 = s[3], x4 = s[4], x5 = s[5], x6 = s[6], x7 = s[7], x8 = s[8],
             x9 = s[9], x10 = s[10], x11 = s[11], x12 = s[12], x13 = s[13], x14 = s[14], x15 = s[15];
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (int r = 0; r < kRounds; r += 2) {
        quarterround(x0, x4, x8, x12);
        quarterround(x1, x5, x9, x13);
        quarterround(x2, x6, x10, x14);
        quarterround(x3, x7, x11, x15);
        quarterround(x0, x5, x10, x15);
        quarterround(x1, x6, x11, x12);
        quarterround(x2, x7, x8, x13);
        quarterround(x3, x4, x9, x14);
    }
    out[0] = x0 + s[0]; out[1] = x1 + s[1]; out[2] = x2 + s[2]; out[3] = x3 + s[3];
    out[4] = x4 + s[4]; out[5] = x5 + s[5]; out[6] = x6 + s[6]; out[7] = x7 + s[7];
    out[8] = x8 + s[8]; out[9] = x9 + s[9]; out[10] = x10 + s[10]; out[11] = x11 + s[11];
    out[12] = x12 + s[12]; out[13] = x13 + s[13]; out[14] = x14 + s[14]; out[15] = x15 + s[15];
}

/* Expand ONE coefficient: uniform in [0, p), addressed by (key, digit, p, slot).
 * n16 = N >> 4 (the escalation stride). Reference path — computes a full block per
 * attempt; the stage-3 kernel path amortizes the base block across 16 lanes. */
FIDESLIB_KSK_HD uint32_t expand_coeff(const uint32_t key[8], uint32_t digit, uint32_t p, uint32_t slot, uint32_t n16) {
    const uint32_t b0 = slot >> 4;
    const uint32_t w = slot & 15u;
    const uint32_t m_p = (0xFFFFFFFFu / p) * p;  // rejection threshold m*p
    uint32_t out[16];
    uint32_t v = 0;
    for (uint32_t t = 0; t < (uint32_t)kTMax; ++t) {
        chacha_block(key, b0 + t * n16, digit, p, out);
        v = out[w];
        if (v < m_p)
            break;  // accepted (falls through to v % p); after kTMax rejects: fallback v % p
    }
    return v % p;
}

}  // namespace FIDESlib::CKKS::kskexpand

#undef FIDESLIB_KSK_HD

#endif  // FIDESLIB_CKKS_KSKSEEDEXPAND_CUH
