/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_profiler_api.h>
#include <mutex>
#include <unordered_map>
#include "cuda/utils//DeviceDefs.cuh"
#include "cuda/utils//DeviceUtils.h"

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/printf.h>
#include <fmt/ranges.h>

#include <absl/log/check.h>
#include <absl/log/log.h>


namespace cann {
namespace gpu {

int getCurrentDevice() {
  int dev = -1;
  CUDA_CHECK(cudaGetDevice(&dev));
  CHECK(dev != -1) << "don't have cuda device";

  return dev;
}

void setCurrentDevice(int device) {
  CUDA_CHECK(cudaSetDevice(device));
}

int getNumDevices() {
  int numDev = -1;
  cudaError_t cuda_error = cudaGetDeviceCount(&numDev);
  if (cudaErrorNoDevice == cuda_error) {
    numDev = 0;
  } else {
    CUDA_CHECK((cuda_error));
  }
  CHECK(numDev != -1) << " dont' have device";

  return numDev;
}

void profilerStart() {
  CUDA_CHECK(cudaProfilerStart());
}

void profilerStop() {
  CUDA_CHECK(cudaProfilerStop());
}

void synchronizeAllDevices() {
  for (int i = 0; i < getNumDevices(); ++i) {
    DeviceScope scope(i);

    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

const cudaDeviceProp& getDeviceProperties(int device) {
  static std::mutex mutex;
  static std::unordered_map<int, cudaDeviceProp> properties;

  std::lock_guard<std::mutex> guard(mutex);

  auto it = properties.find(device);
  if (it == properties.end()) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    properties[device] = prop;
    it = properties.find(device);
  }

  return it->second;
}

const cudaDeviceProp& getCurrentDeviceProperties() {
  return getDeviceProperties(getCurrentDevice());
}

int getMaxThreads(int device) {
  return getDeviceProperties(device).maxThreadsPerBlock;
}

int getMaxThreadsCurrentDevice() {
  return getMaxThreads(getCurrentDevice());
}

dim3 getMaxGrid(int device) {
  auto& prop = getDeviceProperties(device);

  return dim3(prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
}

dim3 getMaxGridCurrentDevice() {
  return getMaxGrid(getCurrentDevice());
}

size_t getMaxSharedMemPerBlock(int device) {
  return getDeviceProperties(device).sharedMemPerBlock;
}

size_t getMaxSharedMemPerBlockCurrentDevice() {
  return getMaxSharedMemPerBlock(getCurrentDevice());
}

int getDeviceForAddress(const void* p) {
  if (!p) {
    return -1;
  }

  cudaPointerAttributes att;
  cudaError_t err = cudaPointerGetAttributes(&att, p);
  CHECK(err == cudaSuccess || err == cudaErrorInvalidValue)
      << fmt::format("unknown error {}", cudaGetErrorString(err));

  if (err == cudaErrorInvalidValue) {
    // Make sure the current thread error status has been reset
    err = cudaGetLastError();
    CHECK(err == cudaErrorInvalidValue) << fmt::format("unknown error {}", cudaGetErrorString(err));
    return -1;
  }

#if USE_AMD_ROCM
  if (att.type != hipMemoryTypeHost && att.type != hipMemoryTypeUnregistered) {
    return att.device;
  } else {
    return -1;
  }
#else
  // memoryType is deprecated for CUDA 10.0+
#if CUDA_VERSION < 10000
  if (att.memoryType == cudaMemoryTypeHost) {
    return -1;
  } else {
    return att.device;
  }
#else
  // FIXME: what to use for managed memory?
  if (att.type == cudaMemoryTypeDevice) {
    return att.device;
  } else {
    return -1;
  }
#endif
#endif
}

bool getFullUnifiedMemSupport(int device) {
  const auto& prop = getDeviceProperties(device);
  return (prop.major >= 6);
}

bool getFullUnifiedMemSupportCurrentDevice() {
  return getFullUnifiedMemSupport(getCurrentDevice());
}

bool getTensorCoreSupport(int device) {
  const auto& prop = getDeviceProperties(device);
  return (prop.major >= 7);
}

bool getTensorCoreSupportCurrentDevice() {
  return getTensorCoreSupport(getCurrentDevice());
}

int getWarpSize(int device) {
  const auto& prop = getDeviceProperties(device);
  return prop.warpSize;
}

int getWarpSizeCurrentDevice() {
  return getWarpSize(getCurrentDevice());
}

size_t getFreeMemory(int device) {
  DeviceScope scope(device);

  size_t free = 0;
  size_t total = 0;

  CUDA_CHECK(cudaMemGetInfo(&free, &total));

  return free;
}

size_t getFreeMemoryCurrentDevice() {
  size_t free = 0;
  size_t total = 0;

  CUDA_CHECK(cudaMemGetInfo(&free, &total));

  return free;
}

DeviceScope::DeviceScope(int device) {
  if (device >= 0) {
    int curDevice = getCurrentDevice();

    if (curDevice != device) {
      prevDevice_ = curDevice;
      setCurrentDevice(device);
      return;
    }
  }

  // Otherwise, we keep the current device
  prevDevice_ = -1;
}

DeviceScope::~DeviceScope() {
  if (prevDevice_ != -1) {
    setCurrentDevice(prevDevice_);
  }
}

CublasHandleScope::CublasHandleScope() {
  auto blasStatus = cublasCreate(&blasHandle_);
  CHECK(blasStatus == CUBLAS_STATUS_SUCCESS) << "cublas error";
}

CublasHandleScope::~CublasHandleScope() {
  auto blasStatus = cublasDestroy(blasHandle_);
  CHECK(blasStatus == CUBLAS_STATUS_SUCCESS) << "cublas error";
}

CudaEvent::CudaEvent(cudaStream_t stream, bool timer) : event_(0) {
  CUDA_CHECK(cudaEventCreateWithFlags(&event_, timer ? cudaEventDefault : cudaEventDisableTiming));
  CUDA_CHECK(cudaEventRecord(event_, stream));
}

CudaEvent::CudaEvent(CudaEvent&& event) noexcept : event_(std::move(event.event_)) {
  event.event_ = 0;
}

CudaEvent::~CudaEvent() {
  if (event_) {
    CUDA_CHECK(cudaEventDestroy(event_));
  }
}

CudaEvent& CudaEvent::operator=(CudaEvent&& event) noexcept {
  event_ = std::move(event.event_);
  event.event_ = 0;

  return *this;
}

void CudaEvent::streamWaitOnEvent(cudaStream_t stream) {
  CUDA_CHECK(cudaStreamWaitEvent(stream, event_, 0));
}

void CudaEvent::cpuWaitOnEvent() {
  CUDA_CHECK(cudaEventSynchronize(event_));
}

} // namespace gpu
} // namespace faiss
