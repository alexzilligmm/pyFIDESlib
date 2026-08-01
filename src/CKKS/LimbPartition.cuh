//
// Created by carlosad on 16/03/24.
//

#ifndef FIDESLIB_CKKS_LIMBPARTITION_CUH
#define FIDESLIB_CKKS_LIMBPARTITION_CUH

#include <atomic>
#include "Limb.cuh"
#include "LimbUtils.cuh"
#include "NTT.cuh"
#include "PeerUtils.cuh"
#ifdef NCCL
#include "nccl.h"
#endif
namespace FIDESlib::CKKS {

extern bool MEMCPY_PEER;
extern bool GRAPH_CAPTURE;
/* Lever 1b-ii: FIDESLIB_KSK_REGEN as a LEVEL (0 off / 1 hoisted / 2 both `a`-readers, the
 * shipping config / 3 hoisted + the wall-negative stage-A smem arm, diagnostic only). ONE
 * definition, because the key-loading path and the launch gates must agree exactly: at >=2
 * nothing reads the `a` rows, which is what makes releasing them (adoptKskASeed) legal. */
int kskRegenLevel();

class LimbPartition {
   public:
    ContextData& cc;
    const uint64_t uid;
    int* level;
    const int id;
    const int device;
#ifdef NCCL
    const ncclComm_t rank;  // For NCCL / RCCL
#else
    const int rank;
#endif
    Stream s;

    std::vector<LimbRecord>& meta;
    std::vector<LimbRecord>& SPECIALmeta;
    const std::vector<int>& digitid;
    std::vector<std::vector<LimbRecord>>& DECOMPmeta;
    std::vector<std::vector<LimbRecord>>& DIGITmeta;
    std::vector<LimbRecord>& GATHERmeta;

    std::vector<LimbImpl> limb;
    std::vector<LimbImpl> SPECIALlimb;
    std::vector<std::vector<LimbImpl>> DECOMPlimb;
    std::vector<std::vector<LimbImpl>> DIGITlimb;
    std::vector<LimbImpl> GATHERlimb;

    void** bufferAUXptrs;
    VectorGPU<void*> limbptr;
    VectorGPU<void*> auxptr;
    VectorGPU<void*> SPECIALlimbptr;
    VectorGPU<void*> SPECIALauxptr;

    std::vector<VectorGPU<void*>> DECOMPlimbptr;
    //  std::vector<VectorGPU<void*>> DECOMPauxptr;
    std::vector<VectorGPU<void*>> DIGITlimbptr;
    // std::vector<VectorGPU<void*>> DIGITauxptr;
    VectorGPU<void*> GATHERptr;
    // n32 speed: concatenation of all digits' DECOMPlimbptr entries (digit-major, slot-minor,
    // i.e. entry (start_d + i) == DECOMPlimbptr[d][i]) so modup can run ONE wide INTT over all
    // source limbs instead of dnum per-digit gy~9 launches (288 blocks = 2.3 blocks/SM on A100;
    // measured 169 GB/s vs 465 GB/s at full width, stream overlap only 1.9x). Subview of
    // bufferAUXptrs (a spare slot of the commented-out DECOMPauxptr region); filled once in
    // generateAllDecompAndDigit alongside DECOMPlimbptr.
    VectorGPU<void*> DECOMPALLptr;

    uint64_t* bufferDECOMPandDIGIT = nullptr;
    uint64_t* bufferSPECIAL = nullptr;
    uint64_t* bufferLIMB = nullptr;
    uint64_t* bufferGATHER = nullptr;
    void* bufferDECOMPandDIGIT_handle = nullptr;
    void* bufferGATHER_handle = nullptr;

    /*
    LimbPartition(LimbPartition && lp) :
        device(lp.device),
        rank(lp.rank),
        meta(lp.meta),
        SPECIALmeta(lp.SPECIALmeta),
        DECOMPmeta(lp.DECOMPmeta),
        limb(std::move(lp.limb)),
        SPECIALlimb(std::move(lp.SPECIALlimb)),
        DECOMPlimb(std::move(lp.DECOMPlimb)),
        limbptr(std::move(lp.limbptr)),
        SPECIALlimbptr(std::move(lp.SPECIALlimbptr)),
        DECOMPlimbptr(std::move(lp.DECOMPlimbptr))
        {}
*/

    LimbPartition(LimbPartition&& l) noexcept;

    LimbPartition(ContextData& cc, const uint64_t& uid, int* level, const int id, bool def_stream = false);

    ~LimbPartition();

    /** RATIONAL RESCALING window base (RR_PLAN milestone (c).2): the GLOBAL primeid held by
     *  limb SLOT 0. Zero on classic prefix chains, where slot k has always held primeid k —
     *  so every `PB(k)` launch base below is literally the old `PARTITION(id, k)` and the
     *  generated code is unchanged. On an RR chain slot k holds primeid pbase + k, with
     *  pbase == cc.windowLo(*level); it moves on every rescale because BOTH window edges do.
     *
     *  Kept as explicit state rather than derived from *level: a rescale rebuilds the limb
     *  vector and the level field is momentarily inconsistent with the storage, and the
     *  keyswitch workspaces (whose level pointer is shared) must not silently re-base. */
    int pbase = 0;
    /** Launch base into C_.primeid_partition for limb SLOT `slot` of THIS partition:
     *  `C_.primeid_flattened[PB(slot) + blockIdx.y]` is the global primeid of slot
     *  `slot + blockIdx.y`. The one place the window offset enters the kernels. */
    int PB(int slot) const;

    Global::Globals* getGlobals();
    void binomialDotProduct(LimbPartition& c1, LimbPartition& c2, const std::vector<const LimbPartition*>& c0s,
                            const std::vector<const LimbPartition*>& c1s, const std::vector<const LimbPartition*>& d0s,
                            const std::vector<const LimbPartition*>& d1s, bool ext);
    void binomialMult(LimbPartition& c1, LimbPartition& c2, const LimbPartition& d0, const LimbPartition& d1,
                      bool extend_ins, bool square);
    void generateLimbToLevel(int new_level);

    enum GENERATION_MODE { AUTOMATIC, SINGLE_BUFFER, DUAL_BUFFER };

    void generate(std::vector<LimbRecord>& records, std::vector<LimbImpl>& limbs, VectorGPU<void*>& ptrs, int pos,
                  VectorGPU<void*>* auxptrs, uint64_t* buffer = nullptr, size_t offset = 0,
                  uint64_t* buffer_aux = nullptr, size_t offset_aux = 0, bool noptr = false, int record_base = 0);

    void generateLimb();

    void generateSpecialLimb(bool zero_out, bool for_communication);

    void add(const LimbPartition& p, const bool ext);
    void add(const LimbPartition& a, const LimbPartition& b, const bool ext_a, const bool ext_b);

    void sub(const LimbPartition& p);

    void multElement(const LimbPartition& p);

    void multPt(const LimbPartition& p);

    void modup(LimbPartition& aux_partition);

    template <ALGO algo = ALGO_SHOUP>
    void moddown(LimbPartition& auxLimbs, bool ntt, bool free_special_limbs);

    void rescale();
    /** RATIONAL RESCALING (RR_PLAN (c).2), defined in RationalRescale.cu: move this
     *  partition's storage from level's window to (level-1)'s by the RR modulus switch —
     *  scale by prod(add) (whose residues on the incoming primes are exactly 0, so the
     *  add-back is a zero-extension), then divide out `drop` one prime at a time in the
     *  CPU reference's order. Rebuilds `limb` (both edges move) and therefore re-uploads the
     *  limbptr/auxptr device tables, and re-bases pbase onto the new window. */
    void rrRescale(const std::vector<int>& drop, const std::vector<int>& add, int new_level);
    /** RATIONAL RESCALING, FUSED KEYSWITCH (RR_PLAN (c).4e): modup's per-digit NTT folded into
     *  the KSK dot (NTT_KSK_DOT / _ACC), so the extended digits never round-trip through DRAM.
     *  `this` is the degree-2 term (EVAL); out0/out1 receive the (b, a) products with their
     *  special limbs, ready for moddown. Replaces modup() + launchFusedDotKSK_2 on RR chains. */
    void rrModupDotKSK(LimbPartition& out0, LimbPartition& out1, const LimbPartition& ksk_a,
                       const LimbPartition& ksk_b, LimbPartition& aux_partition);
    /** Re-upload limbptr/auxptr from the current `limb` vector. Needed whenever the limb
     *  array is rebuilt rather than appended to (the RR rescale is the only such path). */
    void refreshLimbPtrs();
    /** Point this partition's limbptr/auxptr at ANOTHER partition's first `n` limbs, and take
     *  its window base. Used by the RR keyswitch workspace so it supplies digit storage
     *  without the ciphertext's data ever being copied. Device-to-device, `n` pointers. */
    void adoptLimbPtrsFrom(const LimbPartition& src, int n);
    /** PINNED staging for refreshLimbPtrs, allocated lazily and kept for the partition's
     *  lifetime. Exists so the pointer-table upload does not need a stream sync to keep its
     *  source alive — a stack vector forced one, twice per RR rescale. */
    void** pin_stage = nullptr;
    /** Completion of the last pin_stage upload. The buffer is REUSED across calls and a
     *  pinned cudaMemcpyAsync is genuinely asynchronous, so the host must not overwrite it
     *  until the previous copy has drained — waiting on this event is free once it has (the
     *  common case) and correct when it has not. A plain reuse would be a silent data race. */
    cudaEvent_t pin_evt = nullptr;
    /** RR (TO-TRY §2.10a): device table of ONE group's dropped-limb pointers, in drop order —
     *  what NTT_RESCALEK stage 1 indexes. RESCALEK_MAX slots, allocated on first use (so only
     *  RR chains ever pay for it) and kept for the partition's lifetime; the gather that fills
     *  it is stream-ordered, so consecutive groups reuse the same slots safely. */
    void** rr_drop_ptr_ = nullptr;
    void** rr_dropptr();
    /** n32 speed: fused composite DOUBLE prime drop (bit-identical to two rescale() calls,
     * ~half the kernel work). Returns false if the shape doesn't fit — caller must then fall
     * back to the sequential per-prime loop. See the definition for the eligibility rules. */
    bool rescale2();

    void freeSpecialLimbs();

    using OptReference = LimbPartition*;
    using OptConstReference = const LimbPartition*;
    struct NTT_fusion_fields {
        OptReference op2;
        OptConstReference pt;
        OptReference res0;
        OptReference res1;
        OptConstReference kska;
        OptConstReference kskb;
    };

    template <ALGO algo, NTT_MODE mode>
    void ApplyNTT(int batch, LimbPartition::NTT_fusion_fields fields, std::vector<LimbImpl>& limb,
                  VectorGPU<void*>& limbptr, VectorGPU<void*>& auxptr, ContextData& cc, const int primeid_init,
                  const int limbsize = -1, const int slot_start = 0);

    template <ALGO algo = ALGO_SHOUP, NTT_MODE mode = NTT_NONE>
    void NTT(int batch = 1, bool sync = false, NTT_fusion_fields fields = NTT_fusion_fields{});

    struct INTT_fusion_fields {
        OptReference res0;
        OptReference res1;
        OptConstReference kska;
        OptConstReference kskb;
        OptConstReference c0;
        OptConstReference c0tilde;
        OptConstReference c1;
        OptConstReference c1tilde;
    };

    template <ALGO algo, INTT_MODE mode>
    void ApplyINTT(int batch, LimbPartition::INTT_fusion_fields fields, std::vector<LimbImpl>& limb,
                   VectorGPU<void*>& limbptr, VectorGPU<void*>& auxptr, ContextData& cc, const int primeid_init,
                   const int limbsize);

    template <ALGO algo = ALGO_SHOUP, INTT_MODE mode = INTT_NONE>
    void INTT(int batch = 1, bool sync = false, INTT_fusion_fields fields = INTT_fusion_fields{});

    static std::vector<VectorGPU<void*>> generateDecompLimbptr(void** buffer,
                                                               const std::vector<std::vector<LimbRecord>>& DECOMPmeta,
                                                               const int device, int offset);

    void generateAllDecompLimb(uint64_t* pInt, size_t offset);

    void generateAllDigitLimb(uint64_t* pInt, size_t offset, int q_band = -1);

    void copyLimb(const LimbPartition& partition);
    void copySpecialLimb(const LimbPartition& p);

    void generateAllDecompAndDigit(bool iskey, int q_band = -1);
    // Banded key (rotation-key limb pruning): Q-limbs allocated only up to
    // q_band (chain position), digits unused at ct level <= q_band skipped.
    // -1 = full key. Guarded in dotKSK.
    int key_q_band = -1;
    // Lever 1b-i (KSK bit-packing): when >0, this partition holds KEY material whose
    // limbptr/DIGITlimbptr device tables point at key_pack_bits-bit packed streams carved
    // from bufferKSKPACK; the dense DECOMP/DIGIT Limb storage is freed (DECOMPlimb/DIGITlimb
    // cleared). Only the fusedDotKSK_2_/hoistedRotateDotKSK_2_ KSK_PACKED=true arms may read
    // these tables. All-u32 (type==0) single-GPU chains only.
    int key_pack_bits = 0;
    uint64_t* bufferKSKPACK = nullptr;
    size_t bufferKSKPACKbytes = 0;
    void packKeyLimbs(int bits);
    // Lever 1b-ii (in-kernel regen): the 256-bit seed this KEY partition's `a` rows were
    // expanded from (recorded by expandKskADigits). When set — and FIDESLIB_KSK_REGEN=1 —
    // the dot kernels' REGEN arms regenerate kska(digit, p, slot) in registers from this
    // seed (KskSeedExpand.cuh FROZEN SPEC v1, bit-identical to the expanded rows) instead
    // of streaming the `a` half of the key from DRAM.
    uint32_t ksk_seed[8] = {};
    bool ksk_seed_set = false;
    // Lever 1b-ii memory endgame: at FIDESLIB_KSK_REGEN>=2 EVERY reader of this chain's `a`
    // rows regenerates them, so the rows are never materialized — adoptKskASeed() records the
    // seed and releases the storage instead. MEASURED at -7.50 GiB (22105 -> 14425 MiB peak,
    // scripts/mem_ab.sh) — do NOT re-derive this from the `Rotation keys loaded: N ~ XXXXX MB`
    // line, which is a formula assuming dense storage, not an allocation; trusting it once
    // produced a wrong -8.5 GB claim. The device pointer tables survive, holding nullptr, so the host
    // staging code that writes them into digits tables is unchanged; anything that would
    // actually READ them must throw first, which is what this flag is for.
    bool ksk_a_released = false;
    void adoptKskASeed(const std::vector<uint32_t>& seed, int q_band = -1);
    // Lever 1b-ii (load-time expansion): fill this KEY partition's `a` DECOMP/DIGIT limbs
    // on-GPU from the 256-bit seed instead of H2D-copying them (bit-identical by the
    // stage-2 gate; builds the same limbptr mapping loadDecompDigit would).
    void expandKskADigits(const std::vector<uint32_t>& seed);

    void mult1AddMult23Add4(const LimbPartition& partition1, const LimbPartition& partition2,
                            const LimbPartition& partition3, const LimbPartition& partition4);

    void mult1Add2(const LimbPartition& partition1, const LimbPartition& partition2);

    void generateLimbSingleMalloc();
    void generateLimbConstant();

    void loadDecompDigit(const std::vector<std::vector<std::vector<uint64_t>>>& data,
                         const std::vector<std::vector<uint64_t>>& moduli);

    void dotKSK(const LimbPartition& src, const LimbPartition& ksk, const bool inplace = false,
                const LimbPartition* limbsrc = nullptr);

    void multElement(const LimbPartition& partition1, const LimbPartition& partition2);

    void multModupDotKSK(LimbPartition& c1, const LimbPartition& c1tilde, LimbPartition& c0,
                         const LimbPartition& c0tilde, const LimbPartition& ksk_a, const LimbPartition& ksk_b);

    int getLimbSize(int level);
    void automorph(const int index, const int br, LimbPartition* src, bool ext);

    void modupInto(LimbPartition& partition, LimbPartition& partition1);
    void multScalar(std::vector<uint64_t>& vector);
    void squareElement(const LimbPartition& p);
    void binomialSquareFold(LimbPartition& c0_res, const LimbPartition& c2_key_switched_0,
                            const LimbPartition& c2_key_switched_1);
    void addScalar(std::vector<uint64_t>& vector);
    void subScalar(std::vector<uint64_t>& vector);
    void dropLimb();
    void addMult(const LimbPartition& partition, const LimbPartition& partition1);
    void broadcastLimb0();
    /** COMPOSITESCALING ModRaise: CRT-extend the bottom d limbs across ALL current limbs
     *  (call after grow()). qhatinv[k] = (Q0/q_k)^{-1} mod q_k; qhat is the flattened
     *  (Q0/q_k) mod q_i table with stride = current limb count. Single-GPU only. */
    void compositeModRaise(int d, const std::vector<uint64_t>& qhatinv, const std::vector<uint64_t>& qhat);
    void evalLinearWSum(uint32_t n, std::vector<const LimbPartition*> ps, std::vector<uint64_t>& weights);
    void rotateModupDotKSK(LimbPartition& c1, LimbPartition& c0, const LimbPartition& ksk_a,
                           const LimbPartition& ksk_b);
    void squareModupDotKSK(LimbPartition& c1, LimbPartition& c0, const LimbPartition& ksk_a,
                           const LimbPartition& ksk_b);
    void rescaleMGPU();
    void moddownMGPU(LimbPartition& auxLimbs, bool ntt, bool free_special_limbs,
                     const std::vector<uint64_t*>& bufferSpecial_);
    void generatePartialSpecialLimb();
    void dotProductPt(LimbPartition& c1, const std::vector<const LimbPartition*>& c0s,
                      const std::vector<const LimbPartition*>& c1s, const std::vector<const LimbPartition*>& pts,
                      bool ext);

    void generateGatherLimb(bool iskey);
    void dotKSKfusedMGPU(LimbPartition& out2, const LimbPartition& digitSrc, const LimbPartition& ksk_a,
                         const LimbPartition& ksk_b, const LimbPartition& src);
    void fusedHoistRotate(int n, std::vector<int> indexes, std::vector<LimbPartition*>& c0,
                          std::vector<LimbPartition*>& c1, const std::vector<LimbPartition*>& ksk_a,
                          const std::vector<LimbPartition*>& ksk_b, const LimbPartition& src_c0,
                          const LimbPartition& src_c1, bool c0_modup);

    void modup_ksk_moddown_mgpu(LimbPartition& c0, const LimbPartition& ksk_a, const LimbPartition& ksk_b,
                                LimbPartition& auxLimbs1, LimbPartition& auxLimbs2, const bool moddown,
                                const std::vector<uint64_t*>& bufferGather_,
                                const std::vector<uint64_t*>& bufferSpecial_c0,
                                const std::vector<uint64_t*>& bufferSpecial_c1, const std::vector<Stream*>& external_s,
                                std::vector<std::vector<std::vector<std::pair<uint64_t, TimelineSemaphore*>>>>& signal,
                                std::vector<std::atomic_uint64_t*>& thread_stop);
    void broadcastLimb0_mgpu();
    void doubleRescaleMGPU(LimbPartition& partition);
    void scaleByP();

    void modupMGPU(LimbPartition& aux, const std::vector<uint64_t*>& bufferGather_,
                   std::vector<std::atomic_uint64_t*>& thread_stop, std::vector<Stream*>& external_s);

    void multNoModdownEnd(LimbPartition& c0, const LimbPartition& bc0, const LimbPartition& bc1,
                          const LimbPartition& in, const LimbPartition& aux);
    static void multScalarBatchManyToOne(std::vector<LimbPartition*>& parta,
                                         const std::vector<std::vector<unsigned long int>>& vector,
                                         const std::vector<std::vector<unsigned long int>>& vector_shoup, int stride,
                                         double usage);

    static void addScalarBatchManyToOne(std::vector<LimbPartition*>& parta,
                                        const std::vector<std::vector<unsigned long int>>& vector, int stride,
                                        double usage);

    static void multPtBatchManyToOne(std::vector<LimbPartition*>& parta, const std::vector<LimbPartition*>& partb,
                                     int stride, double usage);

    static void addBatchManyToOne(std::vector<LimbPartition*>& parta, const std::vector<LimbPartition*>& partb,
                                  int stride, double usage, bool sub, bool exta, bool extb);

    static void LTdotProductPtBatch(std::vector<LimbPartition*>& out, const std::vector<LimbPartition*>& in,
                                    const std::vector<LimbPartition*>& pt, int bStep, int gStep, int stride,
                                    double usage, bool ext);

    static void fusedHoistedRotateBatch(std::vector<LimbPartition*>& out, const std::vector<LimbPartition*>& in,
                                        const std::vector<LimbPartition*>& ksk_a,
                                        const std::vector<LimbPartition*>& ksk_b, const std::vector<int>& indexes,
                                        int n, int stride, double usage, bool c0_modup);

    Stream& getS() const { return const_cast<LimbPartition*>(this)->s; }
};

}  // namespace FIDESlib::CKKS
#endif  //FIDESLIB_CKKS_LIMBPARTITION_CUH
