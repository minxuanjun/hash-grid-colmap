/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <glog/logging.h>
#include <cuda.h>
#include "impl/cannAssert.h"
///
/// Assertions
///

#if defined(__CUDA_ARCH__) || defined(USE_AMD_ROCM)
#define GPU_CANN_ASSERT(X) assert(X)
#define GPU_CANN_ASSERT_MSG(X, MSG) assert(X)
#define GPU_CANN_ASSERT_FMT(X, FMT, ...) assert(X)
#else
#define GPU_CANN_ASSERT(X) CANN_ASSERT(X)
#define GPU_CANN_ASSERT_MSG(X, MSG) CANN_ASSERT_MSG(X, MSG)
#define GPU_CANN_ASSERT_FMT(X, FMT, ...) CANN_ASSERT_FMT(X, FMT, __VA_ARGS__)
// #define GPU_CANN_ASSERT_FMT(X, FMT, ...) FAISS_ASSERT_FMT(X, FMT, __VA_ARGS)
#endif // __CUDA_ARCH__
