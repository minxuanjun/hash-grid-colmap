add_definitions("-DSIFTGPU_NO_DEVIL")
set(CMAKE_CXX_STANDARD 17)

find_package(fmt REQUIRED)

set(OPTIONAL_CUDA_SRCS)
set(OPTIONAL_CUDA_LINK_LIBS)
if(CUDA_ENABLED)
    message(STATUS "Enable CUDA_SIFTGPU_ENABLED")
    add_definitions("-DCUDA_SIFTGPU_ENABLED")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED YES)
    set(CMAKE_CUDA_EXTENSIONS NO)
    if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "10.1")
        message(FATAL_ERROR "CUDA 10.0 and older are not supported. Please upgrade to CUDA 10.1 or newer.")
    endif()
    set(CUDA_SEPARABLE_COMPILATION ON)

    set(OPTIONAL_CUDA_SRCS
        CuTexImage.h CuTexImage.cpp
        ProgramCU.cu
        ProgramCU.h
        PyramidCU.h PyramidCU.cpp
        SiftMatchCU.h SiftMatchCU.cpp
        CANNSiftMatch.cu
        CANNSiftMatch.h CANNSiftMatch.cpp
        cuda/cann.cu
    )
    set(OPTIONAL_CUDA_LINK_LIBS
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
    )
endif()

COLMAP_ADD_LIBRARY(
    NAME colmap_sift_gpu
    SRCS
        FrameBufferObject.h FrameBufferObject.cpp
        GlobalUtil.h GlobalUtil.cpp
        GLTexImage.h GLTexImage.cpp
        ProgramGLSL.h ProgramGLSL.cpp
        ProgramGPU.h
        PyramidGL.h PyramidGL.cpp
        ShaderMan.h ShaderMan.cpp
        SiftGPU.h SiftGPU.cpp
        SiftMatch.h SiftMatch.cpp
        SiftPyramid.h SiftPyramid.cpp
        ${OPTIONAL_CUDA_SRCS}
    PRIVATE_LINK_LIBS
        OpenGL::GL
        GLEW::GLEW
        fmt::fmt
        ${OPTIONAL_CUDA_LINK_LIBS}
)
target_compile_options(colmap_sift_gpu PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>")
target_include_directories(colmap_sift_gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})