/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>
#include "cuda/utils/DeviceDefs.cuh"

namespace cann {
namespace gpu {

// defines to simplify the SASS assembly structure file/line in the profiler
#if CUDA_VERSION >= 9000
#define SHFL_SYNC(VAL, SRC_LANE, WIDTH) \
    __shfl_sync(0xffffffff, VAL, SRC_LANE, WIDTH)
#else
#define SHFL_SYNC(VAL, SRC_LANE, WIDTH) __shfl(VAL, SRC_LANE, WIDTH)
#endif

template <typename T>
inline __device__ T shfl(const T val, int srcLane, int width = kWarpSize) {
#if CUDA_VERSION >= 9000
    return __shfl_sync(0xffffffff, val, srcLane, width);
#else
    return __shfl(val, srcLane, width);
#endif
}

// CUDA SDK does not provide specializations for T*
template <typename T>
inline __device__ T* shfl(T* const val, int srcLane, int width = kWarpSize) {
    static_assert(sizeof(T*) == sizeof(long long), "pointer size");
    long long v = (long long)val;

    return (T*)shfl(v, srcLane, width);
}

template <typename T>
inline __device__ T
shfl_up(const T val, unsigned int delta, int width = kWarpSize) {
#if CUDA_VERSION >= 9000
    return __shfl_up_sync(0xffffffff, val, delta, width);
#else
    return __shfl_up(val, delta, width);
#endif
}

// CUDA SDK does not provide specializations for T*
template <typename T>
inline __device__ T* shfl_up(
        T* const val,
        unsigned int delta,
        int width = kWarpSize) {
    static_assert(sizeof(T*) == sizeof(long long), "pointer size");
    long long v = (long long)val;

    return (T*)shfl_up(v, delta, width);
}

template <typename T>
inline __device__ T
shfl_down(const T val, unsigned int delta, int width = kWarpSize) {
#if CUDA_VERSION >= 9000
    return __shfl_down_sync(0xffffffff, val, delta, width);
#else
    return __shfl_down(val, delta, width);
#endif
}

// CUDA SDK does not provide specializations for T*
template <typename T>
inline __device__ T* shfl_down(
        T* const val,
        unsigned int delta,
        int width = kWarpSize) {
    static_assert(sizeof(T*) == sizeof(long long), "pointer size");
    long long v = (long long)val;
    return (T*)shfl_down(v, delta, width);
}

template <typename T>
inline __device__ T shfl_xor(const T val, int laneMask, int width = kWarpSize) {
#if CUDA_VERSION >= 9000
    return __shfl_xor_sync(0xffffffff, val, laneMask, width);
#else
    return __shfl_xor(val, laneMask, width);
#endif
}

// CUDA SDK does not provide specializations for T*
template <typename T>
inline __device__ T* shfl_xor(
        T* const val,
        int laneMask,
        int width = kWarpSize) {
    static_assert(sizeof(T*) == sizeof(long long), "pointer size");
    long long v = (long long)val;
    return (T*)shfl_xor(v, laneMask, width);
}

#ifdef USE_AMD_ROCM

inline __device__ half shfl(half v, int srcLane, int width = kWarpSize) {
    unsigned int vu = __half2uint_rn(v);
    vu = __shfl(vu, srcLane, width);
    return __uint2half_rn(vu);
}

inline __device__ half shfl_xor(half v, int laneMask, int width = kWarpSize) {
    unsigned int vu = __half2uint_rn(v);
    vu = __shfl_xor(vu, laneMask, width);
    return __uint2half_rn(vu);
}

#else

// CUDA 9.0+ has half shuffle
#if CUDA_VERSION < 9000
inline __device__ half shfl(half v, int srcLane, int width = kWarpSize) {
    unsigned int vu = v.x;
    vu = __shfl(vu, srcLane, width);

    half h;
    h.x = (unsigned short)vu;
    return h;
}

inline __device__ half shfl_xor(half v, int laneMask, int width = kWarpSize) {
    unsigned int vu = v.x;
    vu = __shfl_xor(vu, laneMask, width);

    half h;
    h.x = (unsigned short)vu;
    return h;
}
#endif // CUDA_VERSION

#endif // USE_AMD_ROCM

} // namespace gpu
} // namespace faiss
