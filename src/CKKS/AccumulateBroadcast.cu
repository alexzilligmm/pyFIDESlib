//
// Created by carlosad on 12/06/25.
//

#include "CKKS/AccumulateBroadcast.cuh"

#include "CKKS/Context.cuh"
#include "CKKS/RationalRescale.cuh"
std::vector<int> FIDESlib::CKKS::GetAccumulateRotationIndices(const int bStep, const int stride, const int size) {
    std::vector<int> indices;
    int logbStep = std::bit_width((uint32_t)bStep) - 1;
    for (int s = stride; s < stride * size; s <<= logbStep) {
        for (int idx = s; idx < s * bStep && idx < stride * size; idx += s) {
            indices.push_back(idx);
        }
    }
    return indices;
}
std::vector<int> FIDESlib::CKKS::GetbroadcastRotationIndices(const int bStep, const int initsize, const int outsize) {
    const int size = outsize / initsize;
    const int stride = initsize;
    std::vector<int> indices;

    int logbStep = std::bit_width((uint32_t)bStep) - 1;
    for (int s = stride; s < stride * size; s <<= logbStep) {
        if (stride * s * bStep >= stride * size) {
            for (int i = 1; i <= bStep; ++i) {
                int idx = -(i * s) + 1;
                if (-idx < outsize) {
                    indices.push_back(idx);
                }
            }
        } else {
            for (int idx = s; idx < s * bStep && idx < stride * size; idx += s) {
                indices.push_back(idx);
            }
        }
    }
    return indices;
}

void FIDESlib::CKKS::Accumulate(Ciphertext& ctxt, const int bStep, const int stride, const int size) {
    Context& cc_ = ctxt.cc_;
    ContextData& cc = ctxt.cc;
    // RR_CT_POOL: the rotation targets borrow pooled cts (slots 160+i at the ct's level) —
    // (bStep-1) window constructions per bootstrap were host-issue cost. Contents are fully
    // overwritten by rotate_hoisted (copy / keyswitch-dot + automorph) each round.
    std::vector<Ciphertext> auxOwn;
    std::vector<Ciphertext*> aux(bStep - 1, nullptr);
    if (cc.isRR() && rrCtPool()) {
        for (int i = 0; i < bStep - 1; ++i)
            aux[i] = &rrPooledCiphertext(cc_, ctxt.getLevel(), 160 + i);
    } else {
        auxOwn.reserve(bStep - 1);
        for (int i = 0; i < bStep - 1; ++i) {
            auxOwn.emplace_back(cc_);
            aux[i] = &auxOwn.back();
        }
    }

    int logbStep = std::bit_width((uint32_t)bStep) - 1;
    for (int s = 1; s < size; s <<= logbStep) {
        std::vector<int> indexes;
        std::vector<Ciphertext*> auxptr;
        for (int idx = stride * s; idx < stride * size && idx < bStep * stride * s; idx += stride * s) {
            // std::cout << idx << std::endl;
            indexes.push_back(idx);
            auxptr.emplace_back(aux[idx / stride / s - 1]);
        }
        // RATIONAL RESCALING: the ext=true / extend / modDown trio defers the hoisted keyswitch's
        // ModDown so the adds happen once in the extended P*Q basis — a classic-path optimisation.
        // The RR rotate_hoisted fallback performs FULL rotations (each already moddowned) and only
        // ALLOCATES special limbs for ext=true, never fills them, so the extended add consumed
        // garbage: post-raise finite, pre-CtS nonfinite=64 at slots=64, measured. Under RR do the
        // arithmetic plainly: rotate non-ext, add in Q, no extend/modDown.
        if (ctxt.cc.isRR()) {
            // A ~25%-of-runs corruption was born in this window (post-raise -> pre-CtS decoded
            // all-NaN; correction-independent; GONE under CUDA_LAUNCH_BLOCKING=1 in 10/10 runs
            // and with this entry sync in 10/10) — a stream-ordering hazard between the raise
            // side's writes to the ciphertext and this function's first baby copy. The sync is
            // one device fence per bootstrap (~us against ~100 ms) and makes the boundary a
            // hard ordering point. RR_SYNC_AT overrides for bisection: 0 = off (reproduce the
            // race), 2 adds a fence after the rotation batch. The precise missing stream edge
            // is still to be named — see RR_BTS_RUNLOG [race].
            static const int syncAt = [] {
                const char* e = std::getenv("RR_SYNC_AT");
                return e ? std::atoi(e) : 0;
            }();
            if (syncAt & 1)
                cudaDeviceSynchronize();
            ctxt.rotate_hoisted(indexes, auxptr, false);
            if (syncAt & 2)
                cudaDeviceSynchronize();
            // THE NAMED MISSING EDGE (2026-08-03, closes the RR_BTS_RUNLOG [race] hunt): the
            // adds below write ctxt IN PLACE, and Ciphertext::add waits only ITS operand's
            // stream — so add(aux[0]) could overwrite c0/c1 while the rotations producing
            // aux[1..k] were still READING them. At logN 13 the exposure is 2-3 aux per round
            // (~25 %/run hit rate, the fenced symptom); at logN 16 it is 7 aux x 3 rounds and
            // essentially certain — the entire "logN 16 value bring-up gap" was this. Every
            // rotation's reads of ctxt chain transitively into its RESULT's partition stream
            // (copy/keyswitch-dot/automorph all end joined there), so waiting every aux's
            // partition stream from ctxt's own closes the anti-dependency at event cost.
            // RR_SYNC_AT (now default 0) stays as the sledgehammer diagnostic.
            for (auto* r : auxptr) {
                for (size_t g = 0; g < ctxt.c0.GPU.size(); ++g) {
                    ctxt.c0.GPU[g].s.wait(r->c0.GPU[g].s);
                    ctxt.c0.GPU[g].s.wait(r->c1.GPU[g].s);
                    ctxt.c1.GPU[g].s.wait(r->c0.GPU[g].s);
                    ctxt.c1.GPU[g].s.wait(r->c1.GPU[g].s);
                }
            }
            for (size_t i = 0; i < indexes.size(); ++i) {
                ctxt.add(*auxptr[i]);
                if (syncAt & 4)  // bisection: fence after EACH add
                    cudaDeviceSynchronize();
            }
            // THE ROOT EDGE (named 2026-08-03, closes RUNLOG [race] and the logN 16 gap):
            // LimbPartition::add launches on the PER-LIMB streams and never joins them back
            // into the partition stream — but every reader of this ciphertext (copyLimb, the
            // stage stash, CtS's first baby copy, the next round's clone) waits ONLY the
            // partition stream. The adds above are therefore still in flight when the next
            // consumer reads c0/c1: ~25 %/run at logN 13 (2 rounds, few limbs), ~certain at
            // logN 16 (3 rounds, 26 limbs) — measured 0/5 clean with EVERY in-loop fence
            // combination and 3/3 clean under CUDA_LAUNCH_BLOCKING, which is exactly this
            // signature. Join limb->partition once per round; event cost only.
            for (auto* poly : {&ctxt.c0, &ctxt.c1}) {
                for (auto& g : poly->GPU) {
                    const int ls = g.getLimbSize(poly->getLevel());
                    for (int b = 0; b < ls; b += cc.batch)
                        g.s.wait(STREAM(g.limb[b]));
                }
            }
        } else {
            ctxt.rotate_hoisted(indexes, auxptr, true);
            ctxt.extend();
            for (size_t i = 0; i < indexes.size(); ++i) {
                ctxt.add(*auxptr[i]);
            }
            ctxt.modDown(false);
        }
    }
    if (size * stride == ctxt.slots)
        ctxt.slots = stride;
}

void FIDESlib::CKKS::Broadcast(Ciphertext& ctxt, const int bStep, const int initsize, const int outsize) {

    const int size = outsize / initsize;
    const int stride = initsize;
    Context& cc_ = ctxt.cc_;
    ContextData& cc = ctxt.cc;
    std::vector<Ciphertext> aux;

    for (int i = 0; i < bStep; ++i) {
        aux.emplace_back(cc_);
    }

    int logbStep = std::bit_width((uint32_t)bStep) - 1;
    for (int s = 1; s < size; s <<= logbStep) {
        std::vector<int> indexes;
        std::vector<Ciphertext*> auxptr;
        if (s * bStep >= size) {
            for (int i = 1; i <= bStep; ++i) {
                int idx = -(i * stride * s) + 1;
                if (-idx < outsize) {
                    //  std::cout << idx << std::endl;
                    indexes.push_back(idx);
                    auxptr.emplace_back(&aux[i - 1]);
                }
            }
            ctxt.rotate_hoisted(indexes, auxptr, true);
            ctxt.add(*auxptr[0], *auxptr[1]);
            for (size_t i = 2; i < indexes.size(); ++i) {
                ctxt.add(*auxptr[i]);
            }

        } else {
            for (int idx = stride * s; idx < stride * size && idx < bStep * stride * s; idx += stride * s) {
                //  std::cout << idx << std::endl;
                indexes.push_back(idx);
                auxptr.emplace_back(&aux[idx / stride / s - 1]);
            }
            ctxt.rotate_hoisted(indexes, auxptr, true);
            ctxt.extend();
            for (size_t i = 0; i < indexes.size(); ++i) {
                ctxt.add(*auxptr[i]);
            }
        }
        ctxt.modDown(false);
    }
    if (outsize == ctxt.slots)
        ctxt.slots = initsize;
}