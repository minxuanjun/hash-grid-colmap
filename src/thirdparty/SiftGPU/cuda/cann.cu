#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>
#include <typeindex>
#include "cann_cuda.h"

#include <glog/logging.h>

#include "cuda/utils//DeviceDefs.cuh"
#include "cuda/utils/DeviceUtils.h"
#include "utils/StaticUtils.h"

#include <fmt/compile.h>
#include <fmt/ostream.h>
#include <fmt/printf.h>
#include <fmt/ranges.h>
#include "cuda/utils/WarpSelectKernel.cuh"

namespace cg = cooperative_groups;

__device__ uint32_t combine_hash(uint32_t a, uint32_t b) {
  a ^= b + 0x9e3779b9 + (a << 6) + (a >> 2); // Inspired by boost::hash_combine
  return a;
}

__device__ float4 char42float4(char4 data) {
  return make_float4(data.x, data.y, data.z, data.w);
}

__global__ void generate_hash_key_kernel(
    DeviceDescriptorWrapper dataset,
    HashParam hash_param,
    MatrixWrapper<uint32_t> hash_key,
    MatrixWrapper<uint32_t> hash_key_bucket) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int point_id = blockIdx.x;

  int warp_id = tid / 32;
  int lane_id = tid % 32;

  const float scale = 1.5 / 512.f;

  // load original sift

  char4 load_data = __ldg(&((char4*)(dataset(bid, 0)))[lane_id]);
  float4 orig_sift = make_float4(
      load_data.x * scale, load_data.y * scale, load_data.w * scale, load_data.z * scale);

  bool is_even = warp_id & 1;
  int start_idx = is_even ? hash_param.nhash - 1 : 0;
  int end_idx = is_even ? -1 : hash_param.nhash;
  int step = is_even ? -1 : 1;

  int bucket_idx = start_idx;
  for (int i = 0; i < hash_param.nhash; i += 4) {
// 展开 4 次迭代
#pragma unroll 4
    for (int j = 0; j < 4; ++j) {
      if (bucket_idx < 0 || bucket_idx >= hash_param.nhash)
        break;

      // 加载偏移矩阵（全局内存对齐）
      float4 shift_mat = __ldg(&((float4*)(hash_param.shift_mat.data + bucket_idx * 128))[lane_id]);
      float4 shifted_sift = make_float4(
          shift_mat.x + orig_sift.x,
          shift_mat.y + orig_sift.y,
          shift_mat.z + orig_sift.z,
          shift_mat.w + orig_sift.w);
      char4 value = make_char4(
          ceilf(shifted_sift.x),
          ceilf(shifted_sift.y),
          ceilf(shifted_sift.z),
          ceilf(shifted_sift.w));
      uint32_t hash = *reinterpret_cast<uint32_t*>(&value);

      // 使用 Warp Shuffle 合并哈希值
      for (int offset = 16; offset > 0; offset /= 2) {
        hash = combine_hash(hash, __shfl_xor_sync(0xFFFFFFFF, hash, offset));
      }

      // Lane 0 负责写回结果
      if (lane_id == 0) {
        hash &= (hash_param.kNumBins - 1);
        atomicAdd(&hash_key_bucket(bucket_idx, hash), 1);
        hash_key(warp_id, bucket_idx) = hash;
      }

      bucket_idx += step; // 更新 bucket_idx
    }
  }
}

__global__ void generate_hash_key_kernel_version2(
    DeviceDescriptorWrapper dataset,
    HashParam hash_param,
    MatrixWrapper<uint32_t> hash_key,
    MatrixWrapper<uint32_t> hash_key_bucket) {
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32; // 当前 Warp 的全局 ID
  const int lane_id = threadIdx.x % 32;                             // Warp 内线程 ID

  if (warp_id >= dataset.N)
    return; // 检查是否越界

  const float scale = 1.5 / 512.f;

  // 使用 float4 进行内存对齐加载
  // if (lane_id < dataset.Dim / 4) { // 确保线程不会越界
  char4 load_data = ((char4*)(dataset(warp_id, 0)))[lane_id];
  float4 orig_sift = make_float4(
      load_data.x * scale, load_data.y * scale, load_data.w * scale, load_data.z * scale);
  // }

  // 确定 bucket_idx 遍历方向
  bool is_even = warp_id & 1;
  int start_idx = is_even ? hash_param.nhash - 1 : 0;
  int end_idx = is_even ? -1 : hash_param.nhash;
  int step = is_even ? -1 : 1;

  // 遍历哈希桶

  for (int bucket_idx = start_idx; bucket_idx != end_idx; bucket_idx += step) {
    // 加载偏移矩阵（全局内存对齐）
    float4 shift_mat = ((float4*)(hash_param.shift_mat.data + bucket_idx * 128))[lane_id];
    char4 value = make_char4(
        __float2int_rn(shift_mat.x + orig_sift.x),
        __float2int_rn(shift_mat.y + orig_sift.y),
        __float2int_rn(shift_mat.z + orig_sift.z),
        __float2int_rn(shift_mat.w + orig_sift.w));
    uint32_t hash = *reinterpret_cast<uint32_t*>(&value);

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      hash = combine_hash(hash, __shfl_xor_sync(0xFFFFFFFF, hash, offset));
    }

    // Lane 0 负责写回结果
    if (lane_id == 0) {
      hash &= (hash_param.kNumBins - 1);
      atomicAdd(&hash_key_bucket(bucket_idx, hash), 1);
      hash_key(warp_id, bucket_idx) = hash;
    }
  }
}

__global__ void generate_hash_key_kernel_version3(
    DeviceDescriptorWrapper dataset,
    HashParam hash_param,
    MatrixWrapper<uint32_t> hash_key,
    MatrixWrapper<uint32_t> hash_key_bucket) {
  const int bucket_idx = blockIdx.x; // 每个 block 处理一个哈希桶
  const int thread_id = threadIdx.x; // 每个线程的索引
  const int warp_id = thread_id / 32;
  const int lane_id = thread_id % 32;

  // 检查是否越界
  if (bucket_idx >= hash_param.nhash)
    return;

  // 声明共享内存，用于存储 shift_mat 数据
  __shared__ float4 shared_shift_mat[128]; // 每个哈希桶最多加载 128 个 float4（512 字节）

  // 每个哈希桶加载一次 shift_mat 到共享内存
  if (thread_id < 128) {
    shared_shift_mat[thread_id] =
        ((float4*)(hash_param.shift_mat.data + bucket_idx * 128))[thread_id];
  }
  __syncthreads();

  const float scale = 1.5 / 512.f;

  // 遍历数据集中的所有元素（每个 Warp 负责一部分数据）
  for (int warp_data_id = warp_id; warp_data_id < dataset.N; warp_data_id += blockDim.x / 32) {
    float4 orig_sift = {0, 0, 0, 0};
    // if (lane_id < dataset.Dim / 4) {
    char4 load_data = ((char4*)(dataset.data + warp_data_id * dataset.Dim))[lane_id];
    orig_sift = make_float4(
        load_data.x * scale, load_data.y * scale, load_data.z * scale, load_data.w * scale);
    // }

    // 加载共享内存中的 shift_mat 并计算哈希值
    float4 shift_mat = shared_shift_mat[lane_id];
    float4 shifted_sift = make_float4(
        shift_mat.x + orig_sift.x,
        shift_mat.y + orig_sift.y,
        shift_mat.z + orig_sift.z,
        shift_mat.w + orig_sift.w);
    char4 value = make_char4(
        ceilf(shifted_sift.x), ceilf(shifted_sift.y), ceilf(shifted_sift.z), ceilf(shifted_sift.w));
    uint32_t hash = *reinterpret_cast<uint32_t*>(&value);

    // 使用 Warp Shuffle 合并哈希值
    for (int offset = 16; offset > 0; offset /= 2) {
      hash = combine_hash(hash, __shfl_xor_sync(0xFFFFFFFF, hash, offset));
    }

    // Lane 0 负责写回结果
    if (lane_id == 0) {
      hash &= (hash_param.kNumBins - 1);
      atomicAdd(&hash_key_bucket(bucket_idx, hash), 1);
      hash_key(warp_data_id, bucket_idx) = hash;
    }
  }
}

__global__ void distribute_points_to_buckets(
    DeviceDescriptorWrapper dataset,
    HashParam hash_param,
    MatrixWrapper<uint32_t> hash_key,
    MatrixWrapper<uint32_t> bucket_counts,
    MatrixWrapper<uint32_t> hash_key_bucket,
    MatrixWrapper<uint32_t> hash_value_bucket) {
  int block_id = blockIdx.x;
  int thread_idx = threadIdx.x;
  int thread_num = blockDim.x;

  for (int point_id = thread_idx; point_id < dataset.N; point_id += thread_num) {
    const int32_t hash = hash_key(point_id, block_id);
    const int32_t local_id = atomicAdd(&(bucket_counts(block_id, hash)), 1);

    int bucket_start = hash_key_bucket(block_id, hash);
    hash_value_bucket(block_id, bucket_start + local_id) = point_id;
  }
}

void generate_hash_key(
    DeviceDescriptorWrapper dataset,
    HashParam hash_param,
    MatrixWrapper<uint32_t> hash_key,
    MatrixWrapper<uint32_t> hash_key_bucket) {
  CHECK(hash_key.height == dataset.N)
      << "hash_key.height " << hash_key.height << " dataset.N: " << dataset.N;
  CHECK(hash_key.width == hash_param.nhash)
      << "hash_key.height " << hash_key.height << " hash_param.nhash: " << hash_param.nhash;

  CHECK(hash_key_bucket.height == hash_param.nhash)
      << "hash_key_bucket.height " << hash_key_bucket.height << " nhash: " << hash_param.nhash;
  CHECK(hash_key_bucket.width == hash_param.kNumBins)
      << "hash_key_bucket: " << hash_key_bucket.width << " != "
      << "hash_param.kNumBins: " << hash_param.kNumBins;

  int block_size = 32;
  int block_num = dataset.N;

  // generate_hash_key_kernel<<<block_num, block_size>>>(
  //     dataset, hash_param, hash_key, hash_key_bucket);

  block_size = 512;
  int point_num_per_block = block_size / 32;
  block_num = (dataset.N + point_num_per_block - 1) / point_num_per_block;
  generate_hash_key_kernel_version2<<<block_num, block_size>>>(
      dataset, hash_param, hash_key, hash_key_bucket);

  // generate_hash_key_kernel_version3<<<hash_param.nhash, 512>>>(dataset, hash_param, hash_key,
  // hash_key_bucket);
  CUDA_CHECK_LAST_ERROR();
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void BlockExclusiveSumKernel(
    int32_t* d_hash_bucket_value, // 输入和输出数组
    int32_t n_size)               // 每个桶的大小
{
  int bucket_idx = blockIdx.x;

  int* bucket = &d_hash_bucket_value[bucket_idx * n_size];

  using BlockScan = cub::BlockScan<int, BLOCK_THREADS>;
  using TempStorage = typename BlockScan::TempStorage;
  __shared__ TempStorage temp_storage;

  int thread_data[ITEMS_PER_THREAD] = {0}; // 初始化线程私有数据为 0
  int thread_offset = threadIdx.x * ITEMS_PER_THREAD;

  // 从全局内存读取数据到线程私有数组
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    int index = thread_offset + i;
    if (index < n_size) {
      thread_data[i] = bucket[index];
    }
  }

  // 调试：打印每个线程的输入数据
  // if(bucket_idx == 0 && threadIdx.x == 0) {
  //   for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
  //     printf("Block %d, Thread %d, Input[%d] = %d\n", bucket_idx, threadIdx.x, i,
  //     thread_data[i]);
  //   }
  // }
  // 计算块级 Exclusive Sum
  // int exclusive_sum_output[ITEMS_PER_THREAD];
  // cub::BlockScan<int, BLOCK_THREADS>(temp_storage).ExclusiveSum(thread_data,
  // exclusive_sum_output);
  cub::BlockScan<int, BLOCK_THREADS>(temp_storage).ExclusiveSum(thread_data, thread_data);

  // 调试：打印每个线程的 Exclusive Sum 结果
  // if (bucket_idx == 0 && threadIdx.x ==0) {
  //   for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
  //     printf("Block %d, Thread %d, Output[%d] = %d\n", bucket_idx, threadIdx.x, i,
  //     thread_data[i]);
  //   }
  // }

  // 将结果写回全局内存
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    int index = thread_offset + i;
    if (index < n_size) {
      bucket[index] = thread_data[i];
    }
  }
}

void build_hash_bucket_indexes(
    DeviceDescriptorWrapper dataset,
    HashParam hash_param,
    MatrixWrapper<uint32_t> hash_key,
    // [N, N_hash]
    MatrixWrapper<uint32_t> hash_key_bucket,
    // [N_hash, bucket_size]
    MatrixWrapper<uint32_t> hash_value_bucket //[N_hash, N]
) {
  CHECK(hash_key.height == dataset.N)
      << "hash_key.height " << hash_key.height << " dataset.N: " << dataset.N;
  CHECK(hash_key.width == hash_param.nhash)
      << "hash_key.height " << hash_key.height << " hash_param.nhash: " << hash_param.nhash;

  CHECK(hash_key_bucket.height == hash_param.nhash)
      << "hash_key_bucket.height " << hash_key_bucket.height << " nhash: " << hash_param.nhash;
  CHECK(hash_key_bucket.width == hash_param.kNumBins)
      << "hash_key_bucket: " << hash_key_bucket.width << " != "
      << "hash_param.kNumBins: " << hash_param.kNumBins;

  CHECK(hash_value_bucket.height == hash_param.nhash)
      << "hash_value_bucket.height " << hash_value_bucket.height << " nhash: " << hash_param.nhash;
  CHECK(hash_value_bucket.width == dataset.N)
      << "hash_value_bucket.height " << hash_value_bucket.height << " dataset.N: " << dataset.N;

  uint32_t* d_hash_key_bucket_tmp_ptr;
  CUDA_CHECK(cudaMalloc(&d_hash_key_bucket_tmp_ptr, hash_key_bucket.width * sizeof(uint32_t)));
  // TODO @pengwang please move below code to one kernel
  for (int bucket_index = 0; bucket_index < hash_param.nhash; bucket_index++) {
    // Exclusive scan hash_key_bucket
    {
      // temporary storage
      void* d_temp_storage = nullptr;
      size_t temp_storage_bytes = 0;
      // calculate temporary storage size
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          d_temp_storage,
          temp_storage_bytes,
          hash_key_bucket.data + bucket_index * hash_key_bucket.width,
          d_hash_key_bucket_tmp_ptr,
          hash_key_bucket.width));

      // Allocate temporary storage
      CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

      // Run exclusive prefix sum
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
          d_temp_storage,
          temp_storage_bytes,
          hash_key_bucket.data + bucket_index * hash_key_bucket.width,
          d_hash_key_bucket_tmp_ptr,
          hash_key_bucket.width));

      CUDA_CHECK(cudaMemcpy(
          hash_key_bucket.data + bucket_index * hash_key_bucket.width,
          d_hash_key_bucket_tmp_ptr,
          hash_key_bucket.width * sizeof(uint32_t),
          cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaFree(d_temp_storage));
    }
  }
  CUDA_CHECK(cudaFree(d_hash_key_bucket_tmp_ptr));

  uint32_t* d_bucket_count_ptr;
  CUDA_CHECK(
      cudaMalloc(&d_bucket_count_ptr, hash_param.nhash * hash_param.kNumBins * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemsetAsync(
      d_bucket_count_ptr, 0, hash_param.nhash * hash_param.kNumBins * sizeof(uint32_t), nullptr));
  MatrixWrapper<uint32_t> bucket_count(hash_param.nhash, hash_param.kNumBins, d_bucket_count_ptr);
  distribute_points_to_buckets<<<hash_param.nhash, 256>>>(
      dataset, hash_param, hash_key, bucket_count, hash_key_bucket, hash_value_bucket);
  CUDA_CHECK(cudaFree(d_bucket_count_ptr));
}

void build_hash_bucket_indexes_version2(
    DeviceDescriptorWrapper dataset,
    HashParam hash_param,
    MatrixWrapper<uint32_t> hash_key,
    // [N, N_hash]
    MatrixWrapper<uint32_t> hash_key_bucket,
    // [N_hash, bucket_size]
    MatrixWrapper<uint32_t> hash_value_bucket //[N_hash, N]
) {
  CHECK(hash_key.height == dataset.N)
      << "hash_key.height " << hash_key.height << " dataset.N: " << dataset.N;
  CHECK(hash_key.width == hash_param.nhash)
      << "hash_key.height " << hash_key.height << " hash_param.nhash: " << hash_param.nhash;

  CHECK(hash_key_bucket.height == hash_param.nhash)
      << "hash_key_bucket.height " << hash_key_bucket.height << " nhash: " << hash_param.nhash;
  CHECK(hash_key_bucket.width == hash_param.kNumBins)
      << "hash_key_bucket: " << hash_key_bucket.width << " != "
      << "hash_param.kNumBins: " << hash_param.kNumBins;

  CHECK(hash_value_bucket.height == hash_param.nhash)
      << "hash_value_bucket.height " << hash_value_bucket.height << " nhash: " << hash_param.nhash;
  CHECK(hash_value_bucket.width == dataset.N)
      << "hash_value_bucket.height " << hash_value_bucket.height << " dataset.N: " << dataset.N;

  uint32_t* d_hash_key_bucket_tmp_ptr;
  CUDA_CHECK(cudaMalloc(&d_hash_key_bucket_tmp_ptr, hash_key_bucket.width * sizeof(uint32_t)));

  const int BLOCK_THREAD = 1024;

  if (hash_key_bucket.width == 1 << 10) {
    const int ITEMS_PER_THREAD = 1;
    BlockExclusiveSumKernel<BLOCK_THREAD, ITEMS_PER_THREAD>
        <<<hash_key_bucket.height, BLOCK_THREAD>>>(
            (int32_t*)(hash_key_bucket.data), hash_key_bucket.width);
    CUDA_CHECK_LAST_ERROR();
  } else if (hash_key_bucket.width == 1 << 11) {
    const int ITEMS_PER_THREAD = 2;
    BlockExclusiveSumKernel<BLOCK_THREAD, ITEMS_PER_THREAD>
        <<<hash_key_bucket.height, BLOCK_THREAD>>>(
            (int32_t*)(hash_key_bucket.data), hash_key_bucket.width);
    CUDA_CHECK_LAST_ERROR();
  } else if (hash_key_bucket.width == 1 << 12) {
    const int ITEMS_PER_THREAD = 4;
    BlockExclusiveSumKernel<BLOCK_THREAD, ITEMS_PER_THREAD>
        <<<hash_key_bucket.height, BLOCK_THREAD>>>(
            (int32_t*)(hash_key_bucket.data), hash_key_bucket.width);
    CUDA_CHECK_LAST_ERROR();
  } else if (hash_key_bucket.width == 1 << 13) {
    const int ITEMS_PER_THREAD = 8;
    LOG(INFO) << fmt::format(
        "execute BlockExclusiveSumKernel<{}, {}>  block {}  block_dim: {}",
        BLOCK_THREAD,
        ITEMS_PER_THREAD,
        hash_key_bucket.height,
        hash_key_bucket.width);

    // std::vector<uint32_t> h_hash_key_bucket(hash_key_bucket.height * hash_key_bucket.width);
    // cudaMemcpy(
    //     h_hash_key_bucket.data(),
    //     hash_key_bucket.data,
    //     hash_key_bucket.getSizeInBytes(),
    //     cudaMemcpyDeviceToHost);
    // LOG(INFO) << fmt::format("o {}", h_hash_key_bucket);
    // LOG(INFO) << h_hash_key_bucket[hash_key_bucket.width - 1];
    // LOG(INFO) << h_hash_key_bucket[2 * hash_key_bucket.width - 1];
    // LOG(INFO) << h_hash_key_bucket[3 * hash_key_bucket.width - 1];
    // LOG(INFO) << "max hash collision: "
    //           << *std::max_element(h_hash_key_bucket.begin(), h_hash_key_bucket.end());
    BlockExclusiveSumKernel<BLOCK_THREAD, ITEMS_PER_THREAD>
        <<<hash_key_bucket.height, BLOCK_THREAD>>>(
            (int32_t*)(hash_key_bucket.data), hash_key_bucket.width);
    // cudaMemcpy(
    //     h_hash_key_bucket.data(),
    //     hash_key_bucket.data,
    //     hash_key_bucket.getSizeInBytes(),
    //     cudaMemcpyDeviceToHost);
    // LOG(INFO) << fmt::format("p {}", h_hash_key_bucket);
    //
    // LOG(INFO) << h_hash_key_bucket[hash_key_bucket.width - 1];
    // LOG(INFO) << h_hash_key_bucket[2 * hash_key_bucket.width - 1];
    // LOG(INFO) << h_hash_key_bucket[3 * hash_key_bucket.width - 1];

    CUDA_CHECK_LAST_ERROR();

  } else if (hash_key_bucket.width == 1 << 14) {
    const int ITEMS_PER_THREAD = 16;
    BlockExclusiveSumKernel<BLOCK_THREAD, ITEMS_PER_THREAD>
        <<<hash_key_bucket.height, BLOCK_THREAD>>>(
            (int32_t*)(hash_key_bucket.data), hash_key_bucket.width);
    CUDA_CHECK_LAST_ERROR();
  } else if (hash_key_bucket.width == 1 << 15) {
    const int ITEMS_PER_THREAD = 32;
    BlockExclusiveSumKernel<BLOCK_THREAD, ITEMS_PER_THREAD>
        <<<hash_key_bucket.height, BLOCK_THREAD>>>(
            (int32_t*)(hash_key_bucket.data), hash_key_bucket.width);
    CUDA_CHECK_LAST_ERROR();
  } else {
    LOG(FATAL) << fmt::format(
        "hash_key_bucket.with {} not  in {}",
        hash_key_bucket.width,
        std::vector<int>{1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14, 1 << 15});
  }
  uint32_t* d_bucket_count_ptr;
  CUDA_CHECK(
      cudaMalloc(&d_bucket_count_ptr, hash_param.nhash * hash_param.kNumBins * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemsetAsync(
      d_bucket_count_ptr, 0, hash_param.nhash * hash_param.kNumBins * sizeof(uint32_t), nullptr));
  MatrixWrapper<uint32_t> bucket_count(hash_param.nhash, hash_param.kNumBins, d_bucket_count_ptr);
  distribute_points_to_buckets<<<hash_param.nhash, 256>>>(
      dataset, hash_param, hash_key, bucket_count, hash_key_bucket, hash_value_bucket);
  CUDA_CHECK_LAST_ERROR();
  CUDA_CHECK(cudaFree(d_bucket_count_ptr));
}

__global__ void match_descriptor_kernel(
    DeviceDescriptorWrapper dataset1,
    DeviceDescriptorWrapper dataset2,
    HashParam hash_param,
    MatrixWrapper<uint32_t> hash_key_bucket1,
    MatrixWrapper<uint32_t> hash_value_bucket1,
    MatrixWrapper<uint32_t> hash_key2,
    MatrixWrapper<float> out_k,
    MatrixWrapper<int32_t> out_v) {
  __shared__ float4 query_point[32];

  const float scale = 1.0 / 512;
  int thread_id = threadIdx.x;
  int block_id = blockIdx.x;
  const int query_point_id = block_id;
  {
    char4* query_point_ptr = (char4*)(dataset2(query_point_id, 0));
    query_point[thread_id] = make_float4(
        query_point_ptr[thread_id].x * scale,
        query_point_ptr[thread_id].y * scale,
        query_point_ptr[thread_id].z * scale,
        query_point_ptr[thread_id].w * scale);
  }
  ::cann::gpu::warpFence();

  float k_init = ::cann::gpu::Limits<float>::getMax();
  int v_init = -1;
  cann::gpu::
      WarpSelect<float, int, false, ::cann::gpu::Comparator<float>, 32, 2, ::cann::gpu::kWarpSize>
          heap(k_init, v_init, 2);

  int land_id = thread_id % 32;

  __shared__ float shared_out_k[2];
  __shared__ int shared_out_v[2];

  // __shared__ float group_k[2];
  // __shared__ int group_v[2];
  //
  float group_k[2];
  int group_v[2];

  if (land_id == 0) {
    shared_out_k[0] = k_init;
    shared_out_k[1] = k_init;
    shared_out_v[0] = v_init;
    shared_out_v[1] = v_init;
  }
  ::cann::gpu::warpFence();

  // loop over all bucket
  for (int group = 0; group < hash_param.nhash; group++) {
    // reset thread queue and warp register
    heap.resetThreadQueue();
    heap.resetWarpRegisters();

    // get query point hash and its corresponding bucket index begin and end
    uint32_t query_hash = hash_key2(query_point_id, group);
    uint32_t bucket_index_begin = hash_key_bucket1(group, query_hash);
    uint32_t bucket_index_end = (query_hash >= hash_param.kNumBins - 1)
                                    ? hash_param.kNumBins
                                    : hash_key_bucket1(group, query_hash + 1);

    // each thread in warps process candidate train point
    int index = thread_id % 32;

    // Whole warps must participate in the selection

    int limit = ::cann::gpu::utils::roundDown(
        bucket_index_end - bucket_index_begin, ::cann::gpu::kWarpSize);
    for (; index < limit; index += cann::gpu::kWarpSize) {
      uint32_t train_point_id = hash_value_bucket1(group, index + bucket_index_begin);

      char4* train_point_ptr = (char4*)(dataset1(train_point_id, 0));
      float distance = 0.f;
#pragma unroll
      for (int j = 0; j < 32; j++) {
        // TODO @peng03.wang

        float4 query_point_j = query_point[j];
        float4 train_point_j = make_float4(
            train_point_ptr[j].x * scale,
            train_point_ptr[j].y * scale,
            train_point_ptr[j].z * scale,
            train_point_ptr[j].w * scale);

        distance = __fmaf_rn(
            query_point_j.x - train_point_j.x, query_point_j.x - train_point_j.x, distance);
        distance = __fmaf_rn(
            query_point_j.y - train_point_j.y, query_point_j.y - train_point_j.y, distance);
        distance = __fmaf_rn(
            query_point_j.z - train_point_j.z, query_point_j.z - train_point_j.z, distance);
        distance = __fmaf_rn(
            query_point_j.w - train_point_j.w, query_point_j.w - train_point_j.w, distance);
      }

      heap.add(distance, train_point_id);
    }

    // Handle non-warp multiple remainder
    if (index + bucket_index_begin < bucket_index_end) {
      uint32_t train_point_id = hash_value_bucket1(group, index + bucket_index_begin);

      char4* train_point_ptr = (char4*)(dataset1(train_point_id, 0));
      float distance = 0.f;
#pragma unroll
      for (int j = 0; j < 32; j++) {
        float4 query_point_j = query_point[j];
        float4 train_point_j = make_float4(
            train_point_ptr[j].x * scale,
            train_point_ptr[j].y * scale,
            train_point_ptr[j].z * scale,
            train_point_ptr[j].w * scale);

        distance = __fmaf_rn(
            query_point_j.x - train_point_j.x, query_point_j.x - train_point_j.x, distance);
        distance = __fmaf_rn(
            query_point_j.y - train_point_j.y, query_point_j.y - train_point_j.y, distance);
        distance = __fmaf_rn(
            query_point_j.z - train_point_j.z, query_point_j.z - train_point_j.z, distance);
        distance = __fmaf_rn(
            query_point_j.w - train_point_j.w, query_point_j.w - train_point_j.w, distance);
      }

      // printf("pt2_id %d , pt1_id %d dist: %f\n", query_point_id, train_point_id, distance);
      heap.addThreadQ(distance, train_point_id);
    }
    heap.reduce();

    // heap.writeOut(group_k, group_v, 2);
    // ::cann::gpu::warpFence();
    group_k[0] = heap.warpK[0];
    group_v[0] = heap.warpV[0];
    group_k[1] = cann::gpu::shfl(heap.warpK[0], 1);
    group_v[1] = cann::gpu::shfl(heap.warpV[0], 1);

    if (land_id == 0) {
    }
    // TODO @pengwang please check this
    if (land_id == 0) {
      if (group_k[0] < shared_out_k[0]) {
        shared_out_k[1] = shared_out_k[0];
        shared_out_v[1] = shared_out_v[0];
        shared_out_k[0] = group_k[0];
        shared_out_v[0] = group_v[0];
      } else if (group_k[0] < shared_out_k[1] && group_v[0] != shared_out_v[0]) {
        shared_out_k[1] = group_k[0];
        shared_out_v[1] = group_v[0];
      }
      if (group_k[1] < shared_out_k[1] && group_v[1] != shared_out_v[0]) {
        shared_out_k[1] = group_k[1];
        shared_out_v[1] = group_v[1];
      }
    }

    ::cann::gpu::warpFence();
  }
  if (thread_id == 0) {
    // if (query_point_id == 0) {
    //   printf(
    //       "write out %d %d %f %f\n",
    //       shared_out_v[0],
    //       shared_out_v[1],
    //       shared_out_k[0],
    //       shared_out_k[1]);
    // }
    out_k(query_point_id, 0) = sqrtf(shared_out_k[0]);
    out_k(query_point_id, 1) = sqrtf(shared_out_k[1]);
    out_v(query_point_id, 0) = shared_out_v[0];
    out_v(query_point_id, 1) = shared_out_v[1];
  }
}

void match_descriptor(
    DeviceDescriptorWrapper dataset1,
    DeviceDescriptorWrapper dataset2,
    HashParam hash_param,
    MatrixWrapper<uint32_t> hash_key_bucket1,
    MatrixWrapper<uint32_t> hash_value_bucket1,
    MatrixWrapper<uint32_t> hash_key2,
    MatrixWrapper<float> out_k,
    MatrixWrapper<int32_t> out_v) {
  CHECK(hash_key_bucket1.height == hash_param.nhash)
      << "hash_key_bucket1.height " << hash_key_bucket1.height << " nhash: " << hash_param.nhash;
  CHECK(hash_key_bucket1.width == hash_param.kNumBins)
      << "hash_key_bucket1: " << hash_key_bucket1.width << " != "
      << "hash_param.kNumBins: " << hash_param.kNumBins;

  CHECK(hash_value_bucket1.height == hash_param.nhash)
      << "hash_value_bucket1.height " << hash_value_bucket1.height
      << " nhash: " << hash_param.nhash;
  CHECK(hash_value_bucket1.width == dataset1.N)
      << "hash_value_bucket1.height " << hash_value_bucket1.height << " dataset1.N: " << dataset1.N;

  CHECK(hash_key2.height == dataset2.N)
      << "hash_key2.height " << hash_key2.height << " dataset.N: " << dataset2.N;
  CHECK(hash_key2.width == hash_param.nhash)
      << "hash_key2.height " << hash_key2.height << " hash_param.nhash: " << hash_param.nhash;

  CHECK(out_k.height == dataset2.N)
      << "out_k.height " << out_k.height << " dataset2.N: " << dataset2.N;
  CHECK(out_v.height == dataset2.N)
      << "out_v.height " << out_v.height << " dataset2.N: " << dataset2.N;
  CHECK(out_k.width == out_v.width && out_k.width == 2)
      << "out_k.width: " << out_k.width << " out_v.width: " << out_v.width << " predict knn: " << 2;

  int block_num = dataset2.N;
  int thread_num = 32;

  match_descriptor_kernel<<<block_num, thread_num>>>(
      dataset1,
      dataset2,
      hash_param,
      hash_key_bucket1,
      hash_value_bucket1,
      hash_key2,
      out_k,
      out_v);
  CUDA_CHECK_LAST_ERROR();
}

__global__ void compute_hash_kernel(uint32_t* input, uint32_t* hash_result) {
  __shared__ uint32_t s_data[128];
  // thread and block indices
  int tx = threadIdx.x;
  int bx = blockIdx.x;

  s_data[tx] = input[bx * 128 + tx];
  __syncthreads();

  // warp-level reduction: reduce with each warp(32 threads)
  uint32_t value = s_data[tx];

  for (int offset = 16; offset > 0; offset /= 2) {
    value ^= __shfl_down_sync(0xFFFFFFFF, value, offset);
  }

  // First thread in each warp writes to shared memory
  if (tx % 32 == 0) {
    s_data[tx / 32] = value;
  }
  __syncthreads();

  // Final reduction by the first warp
  if (tx < 32) {
    value = (tx < 4) ? s_data[tx] : 0;
    for (int offset = 16; offset > 0; offset /= 2) {
      value ^= __shfl_down_sync(0xFFFFFFFF, value, offset);
    }
  }

  if (tx == 0) {
    hash_result[bx] = value;
  }
}