#include "cuda/utils/warpselect/WarpSelectImpl.cuh"

namespace cann{
namespace gpu{
WARP_SELECT_IMPL(float, true, 1,1);
WARP_SELECT_IMPL(float, false, 1,1);

}// namespace gpu
}// namnamespace faiss