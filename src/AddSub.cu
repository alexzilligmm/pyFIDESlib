//
// Created by carlosad on 25/03/24.
//
#include <cassert>
#include "AddSub.cuh"

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

/* Cross-TU launcher: a __global__ TEMPLATE launched from a TU that only sees its declaration
 * gets a weak local stub with no device code. Keep every instantiation here. */
void launchAddBytes(dim3 grid, dim3 block, cudaStream_t stream, void** a, void** b, int primeid_init,
                    int bytes_per_thread) {
    switch (bytes_per_thread) {
        case 16: add_bytes_<16><<<grid, block, 0, stream>>>(a, b, primeid_init); break;
        case 32: add_bytes_<32><<<grid, block, 0, stream>>>(a, b, primeid_init); break;
        default: add_bytes_<16><<<grid, block, 0, stream>>>(a, b, primeid_init); break;
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