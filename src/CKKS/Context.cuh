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
#include <map>
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

    /** RR keyswitch (RR_PLAN (c).4): the dnum*6 digit pointer table, allocated ONCE per
     *  context instead of cudaMallocAsync/FreeAsync per keyswitch, with PINNED host staging so
     *  the upload needs no stream sync. Measured: those two per-call costs were a fifth of the
     *  whole keyswitch at the payload levels, where the window is small and everything else
     *  scales down with it. Freed in ~ContextData.
     *
     *  NOT thread-safe: one table per context, so two threads keyswitching on the same context
     *  would race on it. That is why the classic paths allocate per call. RR is single-threaded
     *  today (RRKeySwitchCore asserts single-GPU); give this a slot pool before that changes. */
     /* 2026-08-03: it was not async-safe SINGLE-threaded either, and this was THE logN 16
     *  bootstrap corruption (RUNLOG [logN16-root]): call i+1's host std::copy overwrote the
     *  pinned table while call i's H2D was pending, and its H2D overwrote the DEVICE table
     *  while call i's dot kernel was still reading it — wrong-but-valid pointers, so
     *  initcheck saw nothing and only fences BETWEEN core calls masked it. Now a RING of
     *  slots with per-slot completion events: reuse waits (in practice never blocks). */
    static constexpr int RR_DIGITS_RING = 8;
    void*** rr_digits_dev = nullptr;   // ring base: RR_DIGITS_RING * dnum*6 entries
    void** rr_digits_host = nullptr;   // pinned ring base, same layout
    cudaEvent_t rr_digits_ev[RR_DIGITS_RING] = {};
    int rr_digits_slot = 0;

    /** RR keyswitch DIGIT workspace (RR_PLAN (c).4): one context-lifetime poly whose
     *  DECOMP/DIGIT arrays are allocated once, instead of every ciphertext growing its own.
     *
     *  MEASURED why: a full payload run cost ~1.19 ms/level against a steady-state 0.49,
     *  because RRKeySwitchCore called generateDecompAndDigit on its INPUT and a fresh
     *  ciphertext per level pays ~dnum*(K+L) limb allocations each time — a cost that scales
     *  with dnum, and inverted the dnum ordering between the two measurements. The classic
     *  path never sees this because its ciphertexts are long-lived.
     *
     *  Nothing is copied: the workspace ADOPTS the input's limb pointers (adoptLimbPtrsFrom),
     *  so it only ever supplies storage for the digits. Single-GPU, single-threaded — one
     *  keyswitch is in flight at a time and its digits are consumed by the dot immediately. */
    std::unique_ptr<RNSPoly> rr_ks_workspace;
    RNSPoly& getRRKeySwitchWorkspace();

    /** RR (TO-TRY §2.10f): LEVEL-KEYED SCRATCH. The classic path reuses one set of temporaries
     *  down a whole circuit, because a classic poly can simply be re-levelled. An RR poly
     *  cannot: its storage IS its window, and every level's window differs in SIZE and in BASE
     *  (`pbase`) — `dropToLevel` throws downward on RR by design, and `generateLimbToLevel`
     *  asserts a live window is never regrown. So a circuit that walks levels was constructing
     *  fresh scratch at every one of them, which MEASURED 1.07 ms of a 5.48 ms 10-level walk
     *  (0.107 ms/level, ~19.5 %) — host time the classic comparator does not carry at all,
     *  since `test_classic_walk` builds its ciphertexts outside its timer.
     *
     *  This is the borrow: one poly per (level, slot), built on first use and held for the
     *  context's lifetime, exactly like rr_ks_workspace above.
     *
     *  CONTRACT: pooled scratch must NOT be rrRescale'd. Its level is its key, so a borrower
     *  that moves it down a level silently poisons the slot for every later borrower — the
     *  accessor therefore CHECKS the level on every hand-out and throws rather than hand back
     *  a poly at the wrong window. Anything that gets rescaled (the running ciphertext, the
     *  c0/c1 of an evalmult) must stay privately owned. */
    std::map<std::pair<int, int>, std::unique_ptr<RNSPoly>> rr_scratch;
    RNSPoly& getRRScratch(int level, int slot);

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

    /** RATIONAL RESCALING (RR_PLAN milestone (c)) — the WINDOW model.
     *
     *  On a classic chain a level l owns the prefix [0, l] of the prime layout, so "level"
     *  and "top limb index" are the same number and slot k always holds global primeid k.
     *  On an RR chain a level owns a contiguous WINDOW [lo(r), hi(r)] of the
     *  inverted-terminal layout (smalls outermost-first, so every rescale is an edge move),
     *  and the level index r is the RESCALE COUNT from the top — limb count does not
     *  identify a level, and BOTH edges move on a rescale.
     *
     *  Everything downstream asks these four helpers instead of assuming the prefix. With an
     *  empty window table they return exactly the prefix answers, so classic chains keep
     *  their old behaviour byte-identically. A poly's slot k holds global primeid
     *  windowLo(level) + k — that offset is LimbPartition::pbase. */
    bool isRR() const { return !param.rrWindows.empty(); }
    int rrNumLevels() const { return (int)param.rrWindows.size() / 2; }
    int windowLo(int level) const {
        if (level < 0)
            return 0;
        return isRR() ? (int)param.rrWindows.at(2 * level) : 0;
    }
    int windowHi(int level) const {
        if (level < 0)
            return -1;
        return isRR() ? (int)param.rrWindows.at(2 * level + 1) : level;
    }
    int windowSize(int level) const { return level < 0 ? 0 : windowHi(level) - windowLo(level) + 1; }
    /** The primes leaving / entering the window on the rescale level -> level-1, in the order
     *  the CPU reference (RRChain::RescaleElement) processes them: left edge first, then
     *  right edge. Only meaningful on an RR chain. */
    void rrRescaleSets(int level, std::vector<int>& drop, std::vector<int>& add) const;

    /** Geometry of ONE keyswitch digit at an RR level (RR_PLAN milestone (c).3).
     *
     *  A digit is `window ∩ global partition`, so unlike the classic prefix chain it is
     *  truncated at BOTH ends. Two facts make it addressable with a pair of offsets:
     *   - its active SOURCE primes are a contiguous run of the partition's global list, and
     *   - its active DESTINATION limbs are the specials followed by ONE contiguous run of the
     *     partition's global destination list (globals-except-this-partition), because the
     *     removed block sits inside [lo, hi] — so the run simply starts lower. */
    struct RRDigitGeom {
        int digit;    //!< global partition index
        int gLo, gHi; //!< the digit's ACTIVE global prime range at this level
        int fromOff;  //!< gLo - (the partition's first global prime): DECOMP slot/prime offset
        int nFrom;    //!< active source count
        int toOff;    //!< DIGIT-list slot shift of the active Q destination run
        int nTo;      //!< nSpecial + active Q destination count
    };
    /** The digits a keyswitch at `level` actually consumes, in ascending partition order. */
    std::vector<RRDigitGeom> rrDigits(int level) const;
    /** RR level of a window given its FIRST modulus and limb count — the GPU twin of
     *  RRChain::RRLevelOfElement, and needed for the same reason: an imported ciphertext
     *  arrives as a bare limb array with no level attached, and under RR the limb count alone
     *  does not identify one. The low edge names the window and the size disambiguates it, so
     *  the pair is unique. Throws naming both if no window matches. */
    int rrLevelOfWindow(uint64_t firstModulus, int nLimbs) const;
    /** Global layout index of a Q prime, by value. The window's low edge, recovered from the
     *  first modulus of an imported limb array. Throws naming the modulus if it is not a Q
     *  prime of this chain. */
    int rrPrimeIndex(uint64_t modulus) const;
    /** Specials at the head of every DIGIT list (single-GPU: K). */
    int rrNumSpecialInDigit() const;
    /** F(level) = prod(dropped) / prod(added) for the rescale level -> level-1: the factor a
     *  ciphertext's SCALE is divided by. The RR analogue of modReduceProduct, and not
     *  expressible by it — an RR rescale adds primes back as well as dropping them, so the
     *  factor is a ratio, and it is keyed by LEVEL rather than by top limb. */
    double rrRescaleFactor(int level) const;
    /** Top level: the whole chain. `L` on a classic chain, the last RR level otherwise. */
    int topLevel() const { return isRR() ? rrNumLevels() - 1 : L; }
    /** The BOTTOM modulus the bootstrap raises from, as a double. Classic: q0. Composite: the
     *  product of the first `compositeDegree` primes. RR: the product of LEVEL 0's WINDOW —
     *  Cheddar's L0 = {q0, 2 tau}, a ~2^78 three-limb modulus on our schedule, and it is NOT
     *  a prefix of the layout (its window is [14, 16]), which is why this cannot be written
     *  as a loop over prime[0..d). */
    double bottomModulus() const {
        double q = 1.0;
        if (isRR())
            for (int i = windowLo(0); i <= windowHi(0); ++i)
                q *= (double)prime.at(i).p;
        else
            for (int i = 0; i < compositeDegree(); ++i)
                q *= (double)prime.at(i).p;
        return q;
    }
    /** Scaling factor at the TOP of the chain — the level a bootstrap raises to. Indexed by
     *  limb on a classic chain and by LEVEL on an RR one (see sfAtLevel for why the two
     *  cannot share an accessor). */
    double sfAtTop() const { return isRR() ? sfAtLevel(topLevel()) : sfAtLimb(L); }

    /** COMPOSITESCALING support (d = primes per CKKS level; 1 on classic chains). */
    int compositeDegree() const { return param.compositeDegree; }
    /** Scaling factor read at a LIMB index. On composite chains OpenFHE stores a SENTINEL
     *  1.0 at every index off the level grid (m_scalingFactorsReal); the import reverses
     *  indices, so the grid condition is (L - limbTop) % d == 0. Reading off-grid is
     *  always a bug — this accessor makes it loud instead of a silent scale of 1. */
    double sfAtLimb(int limbTop) const;
    /** RATIONAL RESCALING scaling factor at RR level `r` (RR_PLAN (c).4c step 1).
     *
     *  Separate from sfAtLimb for two reasons, both of which would otherwise be silent:
     *   - sfAtLimb is keyed by LIMB COUNT, and on an RR chain limb count does not identify a
     *     level (levels drop a variable number of limbs and add some back), so its
     *     composite-grid guard would either abort or hand back a neighbouring level's factor;
     *   - the DIRECTION is opposite. FIDESlib's RR level is the WINDOW index — 0 is the 3-limb
     *     bottom, rrNumLevels()-1 the full top, and rrRescale DECREMENTS it. OpenFHE's level is
     *     the rescale count FROM the top, which is how the RR patch indexes
     *     m_scalingFactorsReal. So r maps to rrNumLevels()-1-r, and getting that backwards
     *     yields a plausible-looking factor from the wrong end of the chain. */
    double sfAtLevel(int r) const;
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