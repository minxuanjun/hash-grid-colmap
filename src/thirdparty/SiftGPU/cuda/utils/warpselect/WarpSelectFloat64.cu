/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cuda/utils/warpselect/WarpSelectImpl.cuh"

namespace cann {
namespace gpu {

WARP_SELECT_IMPL(float, true, 64, 3);
WARP_SELECT_IMPL(float, false, 64, 3);

} // namespace gpu
} // namespace cann
