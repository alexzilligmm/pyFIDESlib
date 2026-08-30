//
// Created by carlos on 6/03/24.
//

#ifndef FIDESLIB_CKKS_CONTEXT_CUH
#define FIDESLIB_CKKS_CONTEXT_CUH

#include "ConstantsGPU.cuh"
#include "LimbUtils.cuh"
#include "Parameters.cuh"
#include "RNSPoly.cuh"

#include <array>
#include <cassert>
#include <iostream>
#include <list>
#include <mutex>
#include <unordered_map>

#ifdef NCCL
#include "nccl.h"
#endif

namespace FIDESlib::CKKS {

struct Precomputations {
    std::vector<Constants> constants;
    std::unique_ptr<Global> globals;
    std::map<int, BootstrapPrecomputation> boot;
    std::vector<RNSPoly> auxPoly;
    std::map<int, RNSPoly> monomialCache;
#ifdef NCCL
    std::map<int, ncclComm_t*> dev_to_communicator;
#endif
    struct KeyPrecomputations {
        std::unique_ptr<KeySwitchingKey> eval_key;
        std::map<int, KeySwitchingKey> rot_keys;
    };
    std::map<KeyHash, KeyPrecomputations> keys;
};

enum RESCALE_TECHNIQUE { NO_RESCALE, FIXEDMANUAL, FIXEDAUTO, FLEXIBLEAUTO, FLEXIBLEAUTOEXT };

extern std::atomic_uint64_t next_uid;

class ContextData {
   public:
    static constexpr const char* loc{"Context"};
    CudaNvtxRange my_range;
    Parameters param;
    Precomputations precom;
    const int logN;
    const int N;
    const RESCALE_TECHNIQUE rescaleTechnique;
    const int& L;
    const int logQ;
    int batch;
    const std::vector<int> GPUid;
    const int& dnum;
    std::vector<std::vector<int>> GPUdigits;
    const std::vector<PrimeRecord>& prime;
    std::vector<std::vector<LimbRecord>> meta;
    const std::vector<int> logQ_d;
    const int& K;
    const int logP;
    const std::vector<PrimeRecord>& specialPrime;

    std::vector<std::vector<LimbRecord>> specialMeta;  // Make const maybe
    std::vector<std::vector<LimbRecord>> splitSpecialMeta;
    std::vector<std::vector<std::vector<LimbRecord>>> decompMeta;  // Make const maybe
    std::vector<std::vector<std::vector<LimbRecord>>> digitMeta;   // Make const maybe
    std::vector<LimbRecord> gatherMeta;

    const std::vector<dim3> limbGPUid;
    const std::vector<int> digitGPUid;

#ifdef NCCL
    ncclUniqueId communicatorID;
    std::vector<ncclComm_t> GPUrank;
#else
    std::vector<int> GPUrank;
#endif

    // V2 (TO-TRY §2.0, 2026-08-01): the keyswitch workspaces are a SLOT POOL. With
    // FIDESLIB_KS_AUX_POOL=2 consecutive keyswitch-bearing ops draw alternating workspace
    // sets, so two independent ops (EvalMod's two Chebyshev branches) stop serializing
    // through the shared aux polys — the nsys stream audit showed their dots convoying
    // back-to-back with the other branch's transforms never co-resident. Pool=1 (default)
    // is byte-identical legacy behaviour. The slot advances ONLY at Ciphertext-level op
    // entries (advanceKsAuxSlot), never mid-op: every getter call within one op must see
    // the same workspace set.
    std::array<std::unique_ptr<RNSPoly>, 2> key_switch_aux = {nullptr};
    std::array<std::unique_ptr<RNSPoly>, 2> key_switch_aux2 = {nullptr};
    std::array<std::unique_ptr<RNSPoly>, 4> moddown_aux = {nullptr};
    int ks_aux_slot = 0;
    std::vector<Stream> top_limb_stream;
    std::vector<uint64_t*> top_limb_buffer;
    std::vector<void*> top_limb_buffer_handle;
    std::vector<VectorGPU<void*>> top_limbptr;

    std::vector<Stream> top_limb_stream2;
    std::vector<uint64_t*> top_limb_buffer2;
    std::vector<void*> top_limb_buffer2_handle;
    std::vector<VectorGPU<void*>> top_limbptr2;

    std::vector<std::vector<Stream>> gatherStream;
    std::vector<std::vector<Stream>> digitStream;
    std::vector<std::vector<std::vector<Stream>>> digitStreamForMemcpyPeer;
    std::vector<std::vector<Stream>> digitStream2;

    // std::vector<RNSPoly> key_switch_digits;
    bool canP2P = false;
    std::list<uint64_t*> free_limb;

    //      std::array<Stream, 8> blockingStream;
    //      std::vector<std::vector<Stream>> asyncStream;
    RNSPoly& getKeySwitchAux();
    RNSPoly& getKeySwitchAux2();
    RNSPoly& getModdownAux(const int num);
    void advanceKsAuxSlot();  // no-op at FIDESLIB_KS_AUX_POOL=1 (the default)

    bool isValidPrimeId(const int i) const;

   public:
    ContextData(const Parameters& param_, const std::vector<int>& devs, const int secBits = 0);
    ~ContextData();

    static int computeLogQ(const int L, std::vector<PrimeRecord>& primes);

    static const int& validateDnum(const std::vector<int>& GPUid, const int& dnum);

    static std::vector<std::vector<LimbRecord>> generateMeta(const std::vector<int>& GPUid, const int dnum,
                                                             const std::vector<std::vector<int>> digitGPUid,
                                                             const std::vector<PrimeRecord>& prime,
                                                             const Parameters& param);

    static std::vector<int> computeLogQ_d(const int dnum, const std::vector<std::vector<LimbRecord>>& meta,
                                          const std::vector<PrimeRecord>& prime);

    static const int& computeK(const std::vector<int>& logQ_d, std::vector<PrimeRecord>& Sprimes, Parameters& param);

    static std::vector<std::vector<LimbRecord>> generateSpecialMeta(const std::vector<std::vector<LimbRecord>>& meta,
                                                                    const std::vector<PrimeRecord>& specialPrime,
                                                                    const int ID0, const std::vector<int>& GPUid);

    static std::vector<std::vector<std::vector<LimbRecord>>> generateDecompMeta(
        const std::vector<std::vector<LimbRecord>>& meta, const std::vector<std::vector<int>> dnum,
        const std::vector<int>& vector, int L);

    static std::vector<std::vector<std::vector<LimbRecord>>> generateDigitMeta(
        const std::vector<std::vector<LimbRecord>>& meta, const std::vector<std::vector<LimbRecord>>& splitSpecialMeta,
        const std::vector<LimbRecord>& specialMeta, const std::vector<std::vector<int>>& digitGPUid,
        const std::vector<int>& GPUid);

    static std::vector<dim3> generateLimbGPUid(const std::vector<std::vector<LimbRecord>>& meta, const int L);

    static std::vector<std::vector<int>> generateGPUdigits(const int dnum, const std::vector<int>& devs);
    static std::vector<std::vector<LimbRecord>> generateSplitSpecialMeta(std::vector<LimbRecord>& specialMeta,
                                                                         const std::vector<int> GPUid);
    static std::vector<LimbRecord> generateGatherMeta(const std::vector<std::vector<LimbRecord>>& meta, int L);

   public:
    std::vector<uint64_t> ElemForEvalMult(int level, const double operand, int level_in = -1);
    std::vector<uint64_t> ElemForEvalAddOrSub(const int level, const double operand, const int noise_deg);
    std::vector<double>& GetCoeffsChebyshev();

    /** Per-call correction-factor override for Bootstrap (armed by the wrapper's
     *  CorrectionScope): -1 = use the per-slots precomputation value. Runtime-only —
     *  nothing precomputed (keys, CtS/StC matrices, levels) depends on the correction
     *  factor; it materializes as the 2^-c raise adjust + the 2^c restore inside ONE
     *  bootstrap call, so mixing values across bootstraps in one execution is safe.
     *  OUT-OF-LINE ACCESSORS ONLY from outside the library: ContextData carries
     *  #ifdef NCCL members, so its field offsets differ between the fideslib build and
     *  consumers compiled without the same define — direct field access from the wrapper
     *  silently reads/writes the wrong offset (cost a GPU-job round-trip to find). */
    void setCorrectionFactorOverride(int cf);
    int getCorrectionFactorOverride() const;
    int correctionFactorOverride = -1;

    /** Bootstrap input pre-scale (armed per call by the wrapper, same discipline as the
     *  correction-factor override): the next Bootstrap multiplies its input by this
     *  factor for FREE — it rides constantEvalMult, an arbitrary double the input is
     *  multiplied by anyway (TO-TRY B15: the PRE side of the range-normalising bootstrap
     *  is free; any restore is the caller's business). 1.0 = neutral. Runtime-only, no
     *  precomputation depends on it. OUT-OF-LINE ACCESSORS ONLY (see above). */
    void setBtsPreScale(double f);
    double getBtsPreScale() const;
    double btsPreScale = 1.0;

    /** COMPOSITESCALING support (d = primes per CKKS level; 1 on classic chains). */
    int compositeDegree() const { return param.compositeDegree; }
    /** Scaling factor read at a LIMB index. On composite chains OpenFHE stores a SENTINEL
     *  1.0 at every index off the level grid (m_scalingFactorsReal); the import reverses
     *  indices, so the grid condition is (L - limbTop) % d == 0. Reading off-grid is
     *  always a bug — this accessor makes it loud instead of a silent scale of 1. */
    double sfAtLimb(int limbTop) const;
    /** Product of the compositeDegree ModReduceFactor entries dropped when rescaling a
     *  ciphertext whose top limb is limbTop (single factor on classic chains). */
    double modReduceProduct(int limbTop) const;
    int GetDoubleAngleIts();
    void AddBootPrecomputation(int slots, BootstrapPrecomputation&& precomp);
    bool HasBootPrecomputation(int slots);
    BootstrapPrecomputation& GetBootPrecomputation(int slots);
    void AddRotationKey(int index, KeySwitchingKey&& ksk);
    KeySwitchingKey& GetRotationKey(int index, const KeyHash& keyID, int slots = -1);
    bool HasRotationKey(int index, const KeyHash& keyID);
    // Erase a single rotation key (frees its GPU limbs via the KeySwitchingKey dtor).
    // Returns true if a key was present and removed. Index is normalized as in AddRotationKey.
    bool RemoveRotationKey(int index, const KeyHash& keyID);
    void AddEvalKey(KeySwitchingKey&& ksk);
    KeySwitchingKey& GetEvalKey(const KeyHash& keyID);
    int GetBootK();
    //int GetBootCorrectionFactor();
    static RESCALE_TECHNIQUE translateRescalingTechnique(lbcrypto::ScalingTechnique technique);
    void PrepareNCCLCommunication();
    const std::vector<int> generateDigitGPUid(std::vector<std::vector<LimbRecord>>& meta, const int L, const int dnum);

    bool hasAuxilarPoly() const;
    RNSPoly getAuxilarPoly();
    void returnAuxilarPoly(RNSPoly&& c);
    void trimAuxilarPoly(size_t size);
    void clearAuxilarPoly();
    void clearAutomorphismKeys(const KeyHash& KeyID = {});
    void clearEvalMultKeys(const KeyHash& KeyID = {});
    void clearBootPrecomputation(int slots = -1);
    void clearParamSwitchKeys(const KeyHash& KeyID = {});

    friend Context GenCryptoContextGPU(const Parameters& param, const std::vector<int>& devs);
    friend void DeregisterCryptoContextGPU(const Parameters& param);
    friend void DeregisterCryptoContextGPU(Context cc);
    friend Context GetCurrentContext();
    friend void SetCurrentContext(Context&);

    /** Memo for ElemForEvalMult: the per-scalar bigint CRT expansion is pure given
     *  (level, level_in, operand) plus context-construction-time state (primes, scaling
     *  factors, compositeDegree), so entries live for the context lifetime with no
     *  invalidation. The operand is keyed on its EXACT bit pattern — the output is a
     *  bit-exact CRT residue vector, any tolerance-matching would silently break the
     *  bit-exactness-vs-OpenFHE property. level_in is normalized (-1 -> level) before
     *  hashing so the two spellings of the same branch share an entry. Appended at the
     *  END of the class: library-internal only (out-of-line-accessor rule above). */
    struct ElemMemoKey {
        int level;
        int level_in;
        uint64_t operand_bits;
        bool operator==(const ElemMemoKey&) const = default;
    };
    struct ElemMemoKeyHash {
        size_t operator()(const ElemMemoKey& k) const {
            uint64_t h = k.operand_bits ^ ((uint64_t(uint32_t(k.level)) << 32) | uint32_t(k.level_in));
            h *= 0x9E3779B97F4A7C15ull;
            return size_t(h ^ (h >> 32));
        }
    };
    std::mutex elem_memo_mutex;
    std::unordered_map<ElemMemoKey, std::vector<uint64_t>, ElemMemoKeyHash> elem_memo;

   public:
    /** FIDESLIB_SCALAR_DEV_MEMO (default off, add.166 add.62) — the DEVICE-side companion to
     *  elem_memo. Returns a persistent device buffer holding ElemForEvalMult's residues, or
     *  nullptr when the knob is off.
     *
     *  Why: LimbPartition::multScalar did cudaMallocAsync + cudaMemcpyAsync FROM PAGEABLE HOST
     *  STACK + cudaFreeAsync on EVERY call (~124/bootstrap, ~71k/token). A sub-64 KB pageable
     *  H2D is staged synchronously by the driver ON THE CALLING HOST THREAD (this repo documents
     *  it at LimbPartition.cu:1543-1545), and with one enqueue thread that host stall IS a GPU
     *  gap — add.166 add.62 measured ~1.0-1.3 ms of real idle per bootstrap.
     *
     *  Why a memo and not a reused staging buffer: the entry is written ONCE at first use and
     *  never again, so there is no write-after-read race against an in-flight async copy. A
     *  single reused pinned buffer would need either a slot ring or a per-call event to be
     *  correct, i.e. it would trade the malloc for the event traffic it was meant to remove.
     *  Keyed identically to elem_memo, so a hit here implies a hit there: same bytes, same
     *  kernel, same order => BIT-IDENTICAL by construction. That is the gate. */
    const uint64_t* DevElemForEvalMult(int level, double operand, int level_in = -1);

    /** FIDESLIB_SCALAR_DEV_MEMO — the same write-once device memo for the ADD side (add.166
     *  add.72). `LimbPartition::addScalar` had the identical per-call cudaMallocAsync + PAGEABLE
     *  cudaMemcpyAsync + cudaFreeAsync triple, and under graph capture that is BOTH a memory node
     *  with an unstable address AND a memcpy node with a dead host source.
     *
     *  ⚠️ `negate` is part of the KEY, not a post-processing step. Ciphertext::addScalar
     *  (Ciphertext.cpp:974-993) computes the residues for |c| and then, for c < 0, rewrites the
     *  vector in place as `p - elem[i]`. A memo returning a shared device buffer cannot be mutated
     *  by its caller, so the flip has to happen inside the builder, which makes the two signs two
     *  distinct entries. Keying on the un-negated operand and flipping afterwards would hand every
     *  caller whichever sign happened to be built first — a silent wrong-answer bug. */
    const uint64_t* DevElemForEvalAddOrSub(int level, double operand, int noise_deg, bool negate);

   private:
    std::unordered_map<ElemMemoKey, uint64_t*, ElemMemoKeyHash> dev_elem_memo;

    struct AddMemoKey {
        int level;
        int noise_deg;
        uint64_t operand_bits;
        bool negate;
        bool operator==(const AddMemoKey&) const = default;
    };
    struct AddMemoKeyHash {
        size_t operator()(const AddMemoKey& k) const {
            uint64_t h = k.operand_bits ^ ((uint64_t(uint32_t(k.level)) << 32) | uint32_t(k.noise_deg));
            h ^= k.negate ? 0x1ull : 0x0ull;
            h *= 0x9E3779B97F4A7C15ull;
            return size_t(h ^ (h >> 32));
        }
    };
    std::unordered_map<AddMemoKey, uint64_t*, AddMemoKeyHash> dev_add_memo;
};

Context GenCryptoContextGPU(const Parameters& param, const std::vector<int>& devs);
void DeregisterCryptoContextGPU(const Parameters& param);
void DeregisterCryptoContextGPU(Context cc);
void DeregisterAllContexts();
Context GetCurrentContext();
void SetCurrentContext(Context& cc);
void AddSecretSwitchingKey(KeySwitchingKey&& ksk_a, KeySwitchingKey&& ksk_b);

bool HasSecretSwitchingKey(const Context& a, const Context& b, const KeyHash& key_b);
KeySwitchingKey& GetSecretSwitchingKey(const Context& a, const Context& b, const KeyHash& key_b);

}  // namespace FIDESlib::CKKS
#endif  //FIDESLIB_CKKS_CONTEXT_CUH