//
// Created by carlosad on 25/03/24.
//
#include <cassert>
#include "AddSub.cuh"
#include <cstdlib>

namespace FIDESlib {

template <typename T>
__global__ void add_(T* a, const T* b, const int primeId) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    //if(threadIdx.x == 0 && blockIdx.x == 0) printf("Prime %d: %lu ", primeId, p_prime);
    //  if(threadIdx.x == 0 && blockIdx.x == 0) printf("Size: %d", blockDim.x * gridDim.x);
    a[idx] = modadd(a[idx], b[idx], primeId);
}

template __global__ void add_(uint64_t* a, const uint64_t* b, const int primeId);

template __global__ void add_(uint32_t* a, const uint32_t* b, const int primeId);

template <typename T>
__global__ void sub_(T* a, const T* b, const int primeId) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //if(threadIdx.x == 0 && blockIdx.x == 0) printf("Prime %d: %lu ", primeId, p_prime);
    a[idx] = modsub(a[idx], b[idx], primeId);
}

template __global__ void sub_(uint64_t* a, const uint64_t* b, const int primeId);

template __global__ void sub_(uint32_t* a, const uint32_t* b, const int primeId);

__global__ void add_(void** a, void** b, const int primeid_init) {

    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        ((uint64_t*)a[blockIdx.y])[idx] =
            modadd(((uint64_t*)a[blockIdx.y])[idx], ((uint64_t*)b[blockIdx.y])[idx], primeid);
    } else {
        ((uint32_t*)a[blockIdx.y])[idx] =
            modadd(((uint32_t*)a[blockIdx.y])[idx], ((uint32_t*)b[blockIdx.y])[idx], primeid);
    }
}

// VECTORIZED in-place add — BYTES-indexed, mirroring copy_bytes_ (NOT the retired copy_v4_).
//
// Why bytes and not elements: `add_` is ONE element per thread, the shape the retired copy_ had.
// That cost copies -38 % (copy_ 19.18 us -> 11.91) because "warp count tracked ELEMENT count,
// not bytes", and add_ was never given the same treatment — it measures 0.73 TB/s against the
// copy path's 1.20, at DRAM 46 % / SM 26 % of peak, saturating neither, i.e. latency/ILP-bound.
//
// Indexing by BYTES rather than elements is what copy_bytes_ settled on 2026-07-29, and it
// matters here: 4 ELEMENTS/thread would be 16 B on u32 but 32 B on u64, and 32 B/thread measured
// a 15 % REGRESSION on n64. At BYTES=16 a u32 thread does 4 elements and a u64 thread does 2, so
// both widths move the same bytes.
//
// ⚠️ Do NOT raise BYTES on an isolated benchmark. 64 B/thread wins a standalone sweep and was a
// 26 % PRODUCTION regression for copies: grid.y == nlimbs, so wider work means proportionally
// fewer blocks, and fewer blocks take a smaller share of the machine under contention. Identical
// grid shape here, identical trap. Retune only with an in-situ wall A/B.
//
// Grid must be {bytes_per_limb/(BYTES*128), nlimbs}, block 128 — the kernel carries no length,
// so the grid has to cover the limb exactly. Callers pass 0 bytes for non-uniform widths and
// fall back to the scalar kernel.
template <int BYTES>
__global__ void add_bytes_(void** a, void** b, const int primeid_init) {
    if constexpr (BYTES == 8) {
        const int i8 = threadIdx.x + blockIdx.x * blockDim.x;
        const int primeid8 = C_.primeid_flattened[primeid_init + blockIdx.y];
        if (ISU64(primeid8)) {
            ((uint64_t*)a[blockIdx.y])[i8] =
                modadd(((uint64_t*)a[blockIdx.y])[i8], ((const uint64_t*)b[blockIdx.y])[i8], primeid8);
        } else {
            uint2 va = ((uint2*)a[blockIdx.y])[i8];
            const uint2 vb = ((const uint2*)b[blockIdx.y])[i8];
            va.x = modadd(va.x, vb.x, primeid8);
            va.y = modadd(va.y, vb.y, primeid8);
            ((uint2*)a[blockIdx.y])[i8] = va;
        }
        return;
    }
    constexpr int V = BYTES / 16;  // 16-byte chunks per thread
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (ISU64(primeid)) {
#pragma unroll
        for (int q = 0; q < V; ++q) {  // 16 B == 2 x uint64
            ulonglong2 va = ((ulonglong2*)a[blockIdx.y])[V * i + q];
            const ulonglong2 vb = ((const ulonglong2*)b[blockIdx.y])[V * i + q];
            va.x = modadd((uint64_t)va.x, (uint64_t)vb.x, primeid);
            va.y = modadd((uint64_t)va.y, (uint64_t)vb.y, primeid);
            ((ulonglong2*)a[blockIdx.y])[V * i + q] = va;
        }
    } else {
#pragma unroll
        for (int q = 0; q < V; ++q) {  // 16 B == 4 x uint32
            uint4 va = ((uint4*)a[blockIdx.y])[V * i + q];
            const uint4 vb = ((const uint4*)b[blockIdx.y])[V * i + q];
            va.x = modadd((uint32_t)va.x, (uint32_t)vb.x, primeid);
            va.y = modadd((uint32_t)va.y, (uint32_t)vb.y, primeid);
            va.z = modadd((uint32_t)va.z, (uint32_t)vb.z, primeid);
            va.w = modadd((uint32_t)va.w, (uint32_t)vb.w, primeid);
            ((uint4*)a[blockIdx.y])[V * i + q] = va;
        }
    }
}

template <int BYTES>
__global__ void sub_bytes_(void** a, void** b, const int primeid_init) {
    if constexpr (BYTES == 8) {
        const int i8 = threadIdx.x + blockIdx.x * blockDim.x;
        const int primeid8 = C_.primeid_flattened[primeid_init + blockIdx.y];
        if (ISU64(primeid8)) {
            ((uint64_t*)a[blockIdx.y])[i8] =
                modsub(((uint64_t*)a[blockIdx.y])[i8], ((const uint64_t*)b[blockIdx.y])[i8], primeid8);
        } else {
            uint2 va = ((uint2*)a[blockIdx.y])[i8];
            const uint2 vb = ((const uint2*)b[blockIdx.y])[i8];
            va.x = modsub(va.x, vb.x, primeid8);
            va.y = modsub(va.y, vb.y, primeid8);
            ((uint2*)a[blockIdx.y])[i8] = va;
        }
        return;
    }
    constexpr int V = BYTES / 16;
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (ISU64(primeid)) {
#pragma unroll
        for (int q = 0; q < V; ++q) {
            ulonglong2 va = ((ulonglong2*)a[blockIdx.y])[V * i + q];
            const ulonglong2 vb = ((const ulonglong2*)b[blockIdx.y])[V * i + q];
            va.x = modsub((uint64_t)va.x, (uint64_t)vb.x, primeid);
            va.y = modsub((uint64_t)va.y, (uint64_t)vb.y, primeid);
            ((ulonglong2*)a[blockIdx.y])[V * i + q] = va;
        }
    } else {
#pragma unroll
        for (int q = 0; q < V; ++q) {
            uint4 va = ((uint4*)a[blockIdx.y])[V * i + q];
            const uint4 vb = ((const uint4*)b[blockIdx.y])[V * i + q];
            va.x = modsub((uint32_t)va.x, (uint32_t)vb.x, primeid);
            va.y = modsub((uint32_t)va.y, (uint32_t)vb.y, primeid);
            va.z = modsub((uint32_t)va.z, (uint32_t)vb.z, primeid);
            va.w = modsub((uint32_t)va.w, (uint32_t)vb.w, primeid);
            ((uint4*)a[blockIdx.y])[V * i + q] = va;
        }
    }
}

template <int BYTES, bool SUB>
__global__ void scalar_addsub_bytes_(void** a, const uint64_t* b, const int primeid_init) {
    if constexpr (BYTES == 8) {
        const int i8 = threadIdx.x + blockIdx.x * blockDim.x;
        const int primeid8 = C_.primeid_flattened[primeid_init + blockIdx.y];
        if (ISU64(primeid8)) {
            const uint64_t s8 = b[primeid8];
            const uint64_t v8 = ((uint64_t*)a[blockIdx.y])[i8];
            ((uint64_t*)a[blockIdx.y])[i8] = SUB ? modsub(v8, s8, primeid8) : modadd(v8, s8, primeid8);
        } else {
            const uint32_t s8 = (uint32_t)b[primeid8];
            uint2 va = ((uint2*)a[blockIdx.y])[i8];
            va.x = SUB ? modsub(va.x, s8, primeid8) : modadd(va.x, s8, primeid8);
            va.y = SUB ? modsub(va.y, s8, primeid8) : modadd(va.y, s8, primeid8);
            ((uint2*)a[blockIdx.y])[i8] = va;
        }
        return;
    }
    constexpr int V = BYTES / 16;
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (ISU64(primeid)) {
        const uint64_t s = b[primeid];
#pragma unroll
        for (int q = 0; q < V; ++q) {
            ulonglong2 va = ((ulonglong2*)a[blockIdx.y])[V * i + q];
            va.x = SUB ? modsub((uint64_t)va.x, s, primeid) : modadd((uint64_t)va.x, s, primeid);
            va.y = SUB ? modsub((uint64_t)va.y, s, primeid) : modadd((uint64_t)va.y, s, primeid);
            ((ulonglong2*)a[blockIdx.y])[V * i + q] = va;
        }
    } else {
        const uint32_t s = (uint32_t)b[primeid];
#pragma unroll
        for (int q = 0; q < V; ++q) {
            uint4 va = ((uint4*)a[blockIdx.y])[V * i + q];
            va.x = SUB ? modsub((uint32_t)va.x, s, primeid) : modadd((uint32_t)va.x, s, primeid);
            va.y = SUB ? modsub((uint32_t)va.y, s, primeid) : modadd((uint32_t)va.y, s, primeid);
            va.z = SUB ? modsub((uint32_t)va.z, s, primeid) : modadd((uint32_t)va.z, s, primeid);
            va.w = SUB ? modsub((uint32_t)va.w, s, primeid) : modadd((uint32_t)va.w, s, primeid);
            ((uint4*)a[blockIdx.y])[V * i + q] = va;
        }
    }
}

/* FIDESLIB_ADD_BYTES selects bytes-per-thread for the whole vectorized pointwise family
 * (16/32/64, default 16), expressed in BYTES so it retunes cleanly on other hardware — the same
 * convention FIDESLIB_COPY_BYTES uses.
 *
 * WHY IT IS A KNOB AND NOT A CONSTANT. The copy path settled on 16 because 64 B/thread won an
 * ISOLATED sweep and was a 26 % PRODUCTION regression (grid.y == nlimbs, so wider work gives
 * proportionally fewer blocks, and fewer blocks take a smaller share of the machine under
 * contention). That verdict was measured on COPIES, which carry no arithmetic to hide latency —
 * these kernels do (a modadd/modmult per element), so the balance is not obviously the same and
 * is worth re-deriving here. Retune ONLY with an in-situ wall A/B (`scripts/level_ab.sh`), never
 * with an isolated kernel sweep. */
int fideslibAddBytes() {
    static const int v = [] {
        const char* e = std::getenv("FIDESLIB_ADD_BYTES");
        const int x = (e && *e) ? std::atoi(e) : 16;
        // 12 B is deliberately absent: it does not TILE the limb (262144/(12*128) = 170.67) and
        // these kernels carry no length, so the grid must cover the limb exactly.
        return (x == 8 || x == 16 || x == 32 || x == 64) ? x : 16;
    }();
    return v;
}

/* Cross-TU launchers: a __global__ TEMPLATE launched from a TU that only sees its declaration
 * gets a weak local stub with no device code. Keep every instantiation here. */
void launchAddBytes(dim3 grid, dim3 block, cudaStream_t stream, void** a, void** b, int primeid_init,
                    int bytes_per_thread) {
    switch (bytes_per_thread) {
        case 8: add_bytes_<8><<<grid, block, 0, stream>>>(a, b, primeid_init); break;
        case 32: add_bytes_<32><<<grid, block, 0, stream>>>(a, b, primeid_init); break;
        case 64: add_bytes_<64><<<grid, block, 0, stream>>>(a, b, primeid_init); break;
        default: add_bytes_<16><<<grid, block, 0, stream>>>(a, b, primeid_init); break;
    }
}

void launchSubBytes(dim3 grid, dim3 block, cudaStream_t stream, void** a, void** b, int primeid_init,
                    int bytes_per_thread) {
    switch (bytes_per_thread) {
        case 8: sub_bytes_<8><<<grid, block, 0, stream>>>(a, b, primeid_init); break;
        case 32: sub_bytes_<32><<<grid, block, 0, stream>>>(a, b, primeid_init); break;
        case 64: sub_bytes_<64><<<grid, block, 0, stream>>>(a, b, primeid_init); break;
        default: sub_bytes_<16><<<grid, block, 0, stream>>>(a, b, primeid_init); break;
    }
}

void launchScalarAddSubBytes(dim3 grid, dim3 block, cudaStream_t stream, void** a, const uint64_t* b,
                             int primeid_init, int bytes_per_thread, bool sub) {
    if (sub) {
        switch (bytes_per_thread) {
            case 8: scalar_addsub_bytes_<8, true><<<grid, block, 0, stream>>>(a, b, primeid_init); break;
            case 32: scalar_addsub_bytes_<32, true><<<grid, block, 0, stream>>>(a, b, primeid_init); break;
            case 64: scalar_addsub_bytes_<64, true><<<grid, block, 0, stream>>>(a, b, primeid_init); break;
            default: scalar_addsub_bytes_<16, true><<<grid, block, 0, stream>>>(a, b, primeid_init); break;
        }
    } else {
        switch (bytes_per_thread) {
            case 8: scalar_addsub_bytes_<8, false><<<grid, block, 0, stream>>>(a, b, primeid_init); break;
            case 32: scalar_addsub_bytes_<32, false><<<grid, block, 0, stream>>>(a, b, primeid_init); break;
            case 64: scalar_addsub_bytes_<64, false><<<grid, block, 0, stream>>>(a, b, primeid_init); break;
            default: scalar_addsub_bytes_<16, false><<<grid, block, 0, stream>>>(a, b, primeid_init); break;
        }
    }
}

__global__ void sub_(void** a, void** b, const int primeid_init) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        ((uint64_t*)a[blockIdx.y])[idx] =
            modsub(((uint64_t*)a[blockIdx.y])[idx], ((uint64_t*)b[blockIdx.y])[idx], primeid);
    } else {
        ((uint32_t*)a[blockIdx.y])[idx] =
            modsub(((uint32_t*)a[blockIdx.y])[idx], ((uint32_t*)b[blockIdx.y])[idx], primeid);
    }
}

__global__ void add_(void** a, void** b, void** c, const int primeid_init) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        ((uint64_t*)a[blockIdx.y])[idx] =
            modadd(((uint64_t*)b[blockIdx.y])[idx], ((uint64_t*)c[blockIdx.y])[idx], primeid);
    } else {
        ((uint32_t*)a[blockIdx.y])[idx] =
            modadd(((uint32_t*)b[blockIdx.y])[idx], ((uint32_t*)c[blockIdx.y])[idx], primeid);
    }
}

__global__ void sub_(void** a, void** b, void** c, const int primeid_init) {
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ISU64(primeid)) {
        ((uint64_t*)a[blockIdx.y])[idx] =
            modsub(((uint64_t*)b[blockIdx.y])[idx], ((uint64_t*)c[blockIdx.y])[idx], primeid);
    } else {
        ((uint32_t*)a[blockIdx.y])[idx] =
            modsub(((uint32_t*)b[blockIdx.y])[idx], ((uint32_t*)c[blockIdx.y])[idx], primeid);
    }
}

__global__ void scalar_add_(void** a, uint64_t* b, const int primeid_init) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];
    
    if (ISU64(primeid)) {
        ((uint64_t*)a[blockIdx.y])[idx] = modadd(((uint64_t*)a[blockIdx.y])[idx], b[primeid], primeid);
    } else {
        ((uint32_t*)a[blockIdx.y])[idx] = modadd(((uint32_t*)a[blockIdx.y])[idx], (uint32_t)b[primeid], primeid);
    }
}

__global__ void scalar_sub_(void** a, uint64_t* b, const int primeid_init) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = C_.primeid_flattened[primeid_init + blockIdx.y];

    if (ISU64(primeid)) {
        ((uint64_t*)a[blockIdx.y])[idx] = modsub(((uint64_t*)a[blockIdx.y])[idx], b[primeid], primeid);
    } else {
        ((uint32_t*)a[blockIdx.y])[idx] = modsub(((uint32_t*)a[blockIdx.y])[idx], (uint32_t)b[primeid], primeid);
    }
}

}  // namespace FIDESlib