//
// Created by minxuan on 11/10/24.
//
#pragma once

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#ifdef __CUDACC__
#define CUDA_UNIVERSAL_QUALIFIER __host__ __device__
#else
#define CUDA_UNIVERSAL_QUALIFIER
#endif

struct DeviceDescriptorWrapper {
  int N;
  int Dim;
  int8_t* data;

  CUDA_UNIVERSAL_QUALIFIER const int8_t* operator()(int i, int j) const {
    return data + i * Dim + j;
  }

  CUDA_UNIVERSAL_QUALIFIER int8_t* operator()(int i, int j) {
    return data + i * Dim + j;
  }
};

template <typename T>
struct MatrixWrapper {
  using Type = T;

  int height = 0;
  int width = 0;
  size_t pitch = 0;
  T* data = nullptr;

  CUDA_UNIVERSAL_QUALIFIER inline T& operator()(int i, int j) {
    return *(reinterpret_cast<T*>(reinterpret_cast<char*>(data) + i * pitch) +
             j);
  }

  CUDA_UNIVERSAL_QUALIFIER inline const T& operator()(int i, int j) const {
    return *(reinterpret_cast<T*>(reinterpret_cast<char*>(data) + i * pitch) +
             j);
  }

  MatrixWrapper(int H, int W, T* data = nullptr)
      : height(H), width(W), data(data) {
    pitch = sizeof(T) *
            width;  // init pitch, will be adjusted later if use cudaMallocPitch
  }

  MatrixWrapper(int H, int W, int pitch, T* data = nullptr)
      : height(H), width(W), pitch(pitch), data(data) {}

  MatrixWrapper() = default;

  CUDA_UNIVERSAL_QUALIFIER inline size_t getSizeInBytes() const {
    return height * pitch;
  }
};

struct HashParam {
  int nhash=16;
  int kNumBins = 1 << 13;
  MatrixWrapper<float> shift_mat;
};

void generate_hash_key(DeviceDescriptorWrapper dataset,
                       HashParam hash_param,
                       MatrixWrapper<uint32_t> hash_key,
                       MatrixWrapper<uint32_t> hash_key_bucket);

void build_hash_bucket_indexes_version2(
    DeviceDescriptorWrapper dataset,
    HashParam hash_param,
    MatrixWrapper<uint32_t> hash_key,
    // [N, N_hash]
    MatrixWrapper<uint32_t> hash_key_bucket,
    // [N_hash, bucket_size]
    MatrixWrapper<uint32_t> hash_value_bucket  //[N_hash, N]
);

void match_descriptor(DeviceDescriptorWrapper dataset1,
                      DeviceDescriptorWrapper dataset2,
                      HashParam hash_param,
                      MatrixWrapper<uint32_t> hash_key_bucket1,
                      MatrixWrapper<uint32_t> hash_value_bucket1,
                      MatrixWrapper<uint32_t> hash_key2,
                      MatrixWrapper<float> out_k,
                      MatrixWrapper<int32_t> out_v);