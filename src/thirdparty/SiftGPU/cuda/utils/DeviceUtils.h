/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
/// Wrapper to test return status of CUDA functions
#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
      std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                << cudaGetErrorString(err) << std::endl;                   \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

#define CUDA_CHECK_LAST_ERROR()                                                                   \
  do {                                                                                            \
    cudaError_t err = cudaGetLastError();                                                         \
    if (err != cudaSuccess) {                                                                     \
      fprintf(                                                                                    \
          stderr, "CUDA error: %s in file %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                                         \
    }                                                                                             \
  } while (0)

namespace cann {
namespace gpu {

/// Returns the current thread-local GPU device
int getCurrentDevice();

/// Sets the current thread-local GPU device
void setCurrentDevice(int device);

/// Returns the number of available GPU devices
int getNumDevices();

/// Starts the CUDA profiler (exposed via SWIG)
void profilerStart();

/// Stops the CUDA profiler (exposed via SWIG)
void profilerStop();

/// Synchronizes the CPU against all devices (equivalent to
/// cudaDeviceSynchronize for each device)
void synchronizeAllDevices();

/// Returns a cached cudaDeviceProp for the given device
const cudaDeviceProp& getDeviceProperties(int device);

/// Returns the cached cudaDeviceProp for the current device
const cudaDeviceProp& getCurrentDeviceProperties();

/// Returns the maximum number of threads available for the given GPU
/// device
int getMaxThreads(int device);

/// Equivalent to getMaxThreads(getCurrentDevice())
int getMaxThreadsCurrentDevice();

/// Returns the maximum grid size for the given GPU device
dim3 getMaxGrid(int device);

/// Equivalent to getMaxGrid(getCurrentDevice())
dim3 getMaxGridCurrentDevice();

/// Returns the maximum smem available for the given GPU device
size_t getMaxSharedMemPerBlock(int device);

/// Equivalent to getMaxSharedMemPerBlock(getCurrentDevice())
size_t getMaxSharedMemPerBlockCurrentDevice();

/// For a given pointer, returns whether or not it is located on
/// a device (deviceId >= 0) or the host (-1).
int getDeviceForAddress(const void* p);

/// Does the given device support full unified memory sharing host
/// memory?
bool getFullUnifiedMemSupport(int device);

/// Equivalent to getFullUnifiedMemSupport(getCurrentDevice())
bool getFullUnifiedMemSupportCurrentDevice();

/// Does the given device support tensor core operations?
bool getTensorCoreSupport(int device);

/// Equivalent to getTensorCoreSupport(getCurrentDevice())
bool getTensorCoreSupportCurrentDevice();

/// Returns the warp size of the given GPU device
int getWarpSize(int device);

/// Equivalent to getWarpSize(getCurrentDevice())
int getWarpSizeCurrentDevice();

/// Returns the amount of currently available memory on the given device
size_t getFreeMemory(int device);

/// Equivalent to getFreeMemory(getCurrentDevice())
size_t getFreeMemoryCurrentDevice();

/// RAII object to set the current device, and restore the previous
/// device upon destruction
class DeviceScope {
 public:
  explicit DeviceScope(int device);
  ~DeviceScope();

 private:
  int prevDevice_;
};

/// RAII object to manage a cublasHandle_t
class CublasHandleScope {
 public:
  CublasHandleScope();
  ~CublasHandleScope();

  cublasHandle_t get() {
    return blasHandle_;
  }

 private:
  cublasHandle_t blasHandle_;
};

// RAII object to manage a cudaEvent_t
class CudaEvent {
 public:
  /// Creates an event and records it in this stream
  explicit CudaEvent(cudaStream_t stream, bool timer = false);
  CudaEvent(const CudaEvent& event) = delete;
  CudaEvent(CudaEvent&& event) noexcept;
  ~CudaEvent();

  inline cudaEvent_t get() {
    return event_;
  }

  /// Wait on this event in this stream
  void streamWaitOnEvent(cudaStream_t stream);

  /// Have the CPU wait for the completion of this event
  void cpuWaitOnEvent();

  CudaEvent& operator=(CudaEvent&& event) noexcept;
  CudaEvent& operator=(CudaEvent& event) = delete;

 private:
  cudaEvent_t event_;
};

// #define FAISS_GPU_SYNC_ERROR 1

#ifdef FAISS_GPU_SYNC_ERROR
#define CUDA_TEST_ERROR()                 \
  do {                                    \
    CUDA_VERIFY(cudaDeviceSynchronize()); \
  } while (0)
#else
#define CUDA_TEST_ERROR()            \
  do {                               \
    CUDA_CHECK(cudaGetLastError()); \
  } while (0)
#endif

/// Call for a collection of streams to wait on
template <typename L1, typename L2>
void streamWaitBase(const L1& listWaiting, const L2& listWaitOn) {
  // For all the streams we are waiting on, create an event
  std::vector<cudaEvent_t> events;
  for (auto& stream : listWaitOn) {
    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(event, stream));
    events.push_back(event);
  }

  // For all the streams that are waiting, issue a wait
  for (auto& stream : listWaiting) {
    for (auto& event : events) {
      CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
    }
  }

  for (auto& event : events) {
    CUDA_CHECK(cudaEventDestroy(event));
  }
}

/// These versions allow usage of initializer_list as arguments, since
/// otherwise {...} doesn't have a type
template <typename L1>
void streamWait(const L1& a, const std::initializer_list<cudaStream_t>& b) {
  streamWaitBase(a, b);
}

template <typename L2>
void streamWait(const std::initializer_list<cudaStream_t>& a, const L2& b) {
  streamWaitBase(a, b);
}

inline void streamWait(
    const std::initializer_list<cudaStream_t>& a, const std::initializer_list<cudaStream_t>& b) {
  streamWaitBase(a, b);
}

} // namespace gpu
} // namespace faiss
