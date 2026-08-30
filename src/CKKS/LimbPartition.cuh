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

    /** Byte count and allocation ROUTE for the buffers handed back to GPUfree.
     *
     *  GPUfree is NOT size-agnostic: it re-derives the free-list bucket from `bytes`, and for
     *  `bytes < 64 K` it rounds up to a power of two and forces caching. So `GPUfree(p, id, 0, …)`
     *  files the block in the **1 KB** bucket of `size_to_memory` — where no allocation of the
     *  real size will ever look for it. That was stranding a 12.58 MB special buffer on every
     *  keyswitch (FAILURE §7; MEASURED at 5.2 GB over one RR walk). The rule is simply that
     *  GPUfree must be given the SAME byte count GPUmalloc was given, which is what
     *  `bufferKSKPACKbytes` already does — these extend that to the other two buffers.
     *
     *  The route matters too: `generateSpecialLimb(for_communication=true)` uses a plain
     *  `cudaMalloc`, and a `cudaMalloc`ed pointer must NOT go to `cudaFreeAsync`. */
    size_t bufferSPECIALbytes = 0;
    bool bufferSPECIALcudaMalloc = false;
    size_t bufferLIMBbytes = 0;
    void* bufferDECOMPandDIGIT_handle = nullptr;
    void* bufferGATHER_handle = nullptr;

    /** FIDESLIB_PERSIST_SCRATCH (default ON, add.166 add.72) — PERSISTENT working memory for the
     *  short-lived device POINTER TABLES that a dozen call sites used to build with a per-call
     *  cudaMallocAsync / cudaMemcpyAsync / cudaFreeAsync triple.
     *
     *  WHY. A captured CUDA graph bakes the addresses its kernel nodes were recorded with. A table
     *  allocated per call gets a DIFFERENT address at replay, and the kernel reads the old one:
     *  that is `fusedDotKSKRegen4_<28>(..., void***, ...)` faulting with "potentially made before
     *  memory is allocated" (add.71, compute-sanitizer). Making the ADDRESS persistent is what
     *  unblocks replay; the per-call memcpy that fills the table stays.
     *
     *  ONE SLOT PER PURPOSE, never shared between purposes — two live tables must not alias.
     *  Grow-only: a bigger request frees the old buffer and allocates a new one, so `bytes` is the
     *  high-water mark, not the current request. Shapes stabilise after a couple of bootstraps.
     *
     *  OWNERSHIP / CONCURRENCY. The slot lives on the object whose stream `s` drives the op — for
     *  the static batch helpers that is element [0] (`parta[0]`, `in[0]`, `out[0]`, `acc0`), the
     *  same object they already borrow the stream from. Reuse across calls is then serialised by
     *  stream ordering, exactly the guarantee the cudaMallocAsync/FreeAsync pair gave. Multi-GPU
     *  runs one partition per omp thread, so partitions never share a slot across threads.
     *
     *  ⚠️ Plain cudaMalloc/cudaFree, NOT GPUmalloc: the pool is gated on power-of-two sizes
     *  (CudaUtils.cu:545) and these sizes are not, so pooling would fall through to the unpooled
     *  path anyway — and the pool's record/wait handshake is the cross-stream cost this removes. */
    enum ScratchSlot {
        SC_BATCH_ADD,
        SC_BATCH_MULTPT,
        SC_BATCH_ADDSCALAR,
        SC_BATCH_MULTSCALAR,
        SC_BATCH_BINOMIAL,
        SC_BATCH_LTDOT,
        SC_BATCH_HOISTROT,
        SC_MGPU_DOTKSK,
        SC_MGPU_HOISTROT,
        SC_MGPU_MODDOWN,
        SC_SCALAR_MULT,
        SC_SCALAR_ADD,
        SC_SCALAR_SUB,
        SC_LINWSUM_W,
        SC_LINWSUM_PS,
        SC_N
    };
    struct DevScratch {
        void* p = nullptr;
        size_t bytes = 0;
    };
    DevScratch scratch_[SC_N];
    /** Buffers superseded by a grow that happened WHILE CAPTURING, where neither the stream sync
     *  nor the free is legal. Drained on the next non-capturing grow and in the destructor. */
    std::vector<void*> scratch_retired_;

    /** Persistent scratch for `slot`, at least `bytes` big. Returns nullptr when the knob is off,
     *  when the allocation fails, or when a GROW would be needed while a graph capture is in
     *  flight — in every case the caller must fall back to its per-call allocation. */
    void* scratchGet(int slot, size_t bytes);
    void scratchFreeAll();

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
                  uint64_t* buffer_aux = nullptr, size_t offset_aux = 0, bool noptr = false);

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
    /** n32 speed: fused composite DOUBLE prime drop (bit-identical to two rescale() calls,
     * ~half the kernel work). Returns false if the shape doesn't fit — caller must then fall
     * back to the sequential per-prime loop. See the definition for the eligibility rules. */
    bool rescale2();

    void freeSpecialLimbs();
    /** Release bufferSPECIAL with the byte count and the allocation route it was created with.
     *  Split out because the destructor needs the identical logic — see FAILURE §7. */
    void freeSpecialBuffer();

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
                  const int limbsize = -1);

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
    // Device-resident operand (FIDESLIB_SCALAR_DEV_MEMO, add.166 add.62). Buffer is owned by
    // ContextData::DevElemForEvalMult and outlives every call.
    void multScalar(const uint64_t* d_elems);
    void squareElement(const LimbPartition& p);
    void binomialSquareFold(LimbPartition& c0_res, const LimbPartition& c2_key_switched_0,
                            const LimbPartition& c2_key_switched_1);
    void addScalar(std::vector<uint64_t>& vector);
    /** FIDESLIB_SCALAR_DEV_MEMO (add.166 add.72) — operand already on device, residues INCLUDING
     *  the sign flip. Buffer is owned by ContextData::DevElemForEvalAddOrSub and outlives the call. */
    void addScalar(const uint64_t* d_elems);
    void subScalar(std::vector<uint64_t>& vector);
    void dropLimb();
    void addMult(const LimbPartition& partition, const LimbPartition& partition1);
    void broadcastLimb0();
    /** COMPOSITESCALING ModRaise: CRT-extend the bottom d limbs across ALL current limbs
     *  (call after grow()). qhatinv[k] = (Q0/q_k)^{-1} mod q_k; qhat is the flattened
     *  (Q0/q_k) mod q_i table with stride = current limb count. Single-GPU only. */
    void compositeModRaise(int d, const std::vector<uint64_t>& qhatinv, const std::vector<uint64_t>& qhat);
    // Centred-aggregate CRT lift for coeff plaintexts (d==2); see coeffLiftCentered2_.
    void coeffLiftCentered(uint64_t q0, uint64_t q1, uint64_t q0inv_mod_q1, uint64_t Qhalf,
                           const std::vector<uint64_t>& Q0_mod_qi);
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

    // acc0 += Σ a0[j]·b0[j]; acc1 += Σ a0[j]·b1[j]+a1[j]·b0[j]; acc2 = Σ a1[j]·b1[j].
    // One binomialMultAccum_ launch per partition (FHE_LANE_BATCH phase 2).
    static void binomialMultAccumBatch(LimbPartition& acc0, LimbPartition& acc1, LimbPartition& acc2,
                                       const std::vector<const LimbPartition*>& a0,
                                       const std::vector<const LimbPartition*>& a1,
                                       const std::vector<const LimbPartition*>& b0,
                                       const std::vector<const LimbPartition*>& b1);

    static void fusedHoistedRotateBatch(std::vector<LimbPartition*>& out, const std::vector<LimbPartition*>& in,
                                        const std::vector<LimbPartition*>& ksk_a,
                                        const std::vector<LimbPartition*>& ksk_b, const std::vector<int>& indexes,
                                        int n, int stride, double usage, bool c0_modup);

    Stream& getS() const { return const_cast<LimbPartition*>(this)->s; }
};

}  // namespace FIDESlib::CKKS
#endif  //FIDESLIB_CKKS_LIMBPARTITION_CUH
