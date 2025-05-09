/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "cuda/utils/Select.cuh"

namespace cann {
namespace gpu {

template <
        typename K,
        typename IndexType,
        bool Dir,
        int NumWarpQ,
        int NumThreadQ,
        int ThreadsPerBlock>
__global__ void warpSelect(
        Tensor<K, 2, true> in,
        Tensor<K, 2, true> outK,
        Tensor<IndexType, 2, true> outV,
        K initK,
        IndexType initV,
        int k) {
    if constexpr ((NumWarpQ == 1 && NumThreadQ == 1) || NumWarpQ >= kWarpSize) {
        constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

        WarpSelect<
                K,
                IndexType,
                Dir,
                Comparator<K>,
                NumWarpQ,
                NumThreadQ,
                ThreadsPerBlock>
                heap(initK, initV, k);

        int warpId = threadIdx.x / kWarpSize;
        idx_t row = idx_t(blockIdx.x) * kNumWarps + warpId;

        if (row >= in.getSize(0)) {
            return;
        }

        idx_t i = getLaneId();
        K* inStart = in[row][i].data();

        // Whole warps must participate in the selection
        idx_t limit = utils::roundDown(in.getSize(1), kWarpSize);

        for (; i < limit; i += kWarpSize) {
            heap.add(*inStart, (IndexType)i);
            inStart += kWarpSize;
        }

        // Handle non-warp multiple remainder
        if (i < in.getSize(1)) {
            heap.addThreadQ(*inStart, (IndexType)i);
        }

        heap.reduce();
        heap.writeOut(outK[row].data(), outV[row].data(), k);
    }
}

void runWarpSelect(
        Tensor<float, 2, true>& in,
        Tensor<float, 2, true>& outKeys,
        Tensor<idx_t, 2, true>& outIndices,
        bool dir,
        int k,
        cudaStream_t stream);

void runWarpSelect(
        Tensor<half, 2, true>& in,
        Tensor<half, 2, true>& outKeys,
        Tensor<idx_t, 2, true>& outIndices,
        bool dir,
        int k,
        cudaStream_t stream);

} // namespace gpu
} // namespace faiss
