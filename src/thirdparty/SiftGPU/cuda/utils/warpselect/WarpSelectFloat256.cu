/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cuda/utils/warpselect/WarpSelectImpl.cuh"

namespace cann {
namespace gpu {

WARP_SELECT_IMPL(float, true, 256, 4);
WARP_SELECT_IMPL(float, false, 256, 4);

} // namespace gpu
} // namespace cann
