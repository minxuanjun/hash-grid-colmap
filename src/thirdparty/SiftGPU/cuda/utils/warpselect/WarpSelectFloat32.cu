
#include "cuda/utils/warpselect/WarpSelectImpl.cuh"

namespace cann{
namespace gpu{
WARP_SELECT_IMPL(float,true, 32,2);
WARP_SELECT_IMPL(float, false, 32,2);
}//namespace gpu
}//namenamespace cann