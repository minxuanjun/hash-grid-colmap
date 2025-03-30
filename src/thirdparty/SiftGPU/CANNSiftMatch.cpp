// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

// self
#include "CANNSiftMatch.h"
#include "cuda/cann_cuda.h"
#include "cuda/utils/DeviceUtils.h"
#include <chrono>
#include <glog/logging.h>
#include <fstream>

SiftMatchCANN::SiftMatchCANN(int max_sift) : SiftMatchGPU() {
  _num_sift[0] = _num_sift[1] = 0;
  _id_sift[0] = -1;
  _id_sift[1] = -1;
  _have_loc[0] = _have_loc[1] = 0;
  __max_sift = max_sift <= 0 ? 4096 : ((max_sift + 31) / 32 * 32);
  _initialized = 0;

  need_update_hash_cache[0] = 1;
  need_update_hash_cache[1] = 1;

  hash_param.nhash = 16;
  hash_param.kNumBins = 1 << 13;
}

SiftMatchCANN::~SiftMatchCANN() noexcept {
  // free sift mat
  CUDA_CHECK(cudaFree(hash_param.shift_mat.data));

  // free descriptor
  CUDA_CHECK(cudaFree(descriptors1_wrapper_.data));
  CUDA_CHECK(cudaFree(descriptors2_wrapper_.data));

  // free hash key1
  CUDA_CHECK(cudaFree(hash_key1.data));
  CUDA_CHECK(cudaFree(hash_key_bucket1.data));
  CUDA_CHECK(cudaFree(hash_value_bucket1.data));

  // free hash key2
  CUDA_CHECK(cudaFree(hash_key2.data));
  CUDA_CHECK(cudaFree(hash_key_bucket2.data));

  // free match result
  CUDA_CHECK(cudaFree(out_dists.data));
  CUDA_CHECK(cudaFree(out_ids.data));
}

void SiftMatchCANN::InitSiftMatch() {
  if (_initialized) return;
  _initialized = 1;
}

bool SiftMatchCANN::Allocate(int max_sift, int mbm) {
  SetMaxSift(max_sift);
  std::cout << "max_sift: " << __max_sift << std::endl;

  if (max_sift <= 10000) {
    hash_param.kNumBins = 1 << 13;
  } else {
    hash_param.kNumBins = 1 << 14;
  }
  //
  Eigen::MatrixXf shift_mat = Eigen::MatrixXf::Random(hash_param.nhash, 128);
  hash_param.shift_mat = MatrixWrapper<float>(hash_param.nhash, 128, nullptr);
  CUDA_CHECK(cudaMalloc((void**)&(hash_param.shift_mat.data),
                        hash_param.shift_mat.getSizeInBytes()));
  CUDA_CHECK(cudaMemcpy(hash_param.shift_mat.data,
                        shift_mat.data(),
                        hash_param.shift_mat.getSizeInBytes(),
                        cudaMemcpyHostToDevice));

  descriptors1_wrapper_.height = __max_sift;
  descriptors1_wrapper_.width = 128;
  descriptors1_wrapper_.pitch = descriptors1_wrapper_.width *
                                sizeof(decltype(descriptors1_wrapper_)::Type);
  CUDA_CHECK(cudaMalloc((void**)&descriptors1_wrapper_.data, __max_sift * 128));

  descriptors2_wrapper_.height = __max_sift;
  descriptors2_wrapper_.width = 128;
  descriptors2_wrapper_.pitch = descriptors2_wrapper_.width *
                                sizeof(decltype(descriptors2_wrapper_)::Type);
  CUDA_CHECK(cudaMalloc((void**)&descriptors2_wrapper_.data, __max_sift * 128));

  hash_key1.height = __max_sift;
  hash_key1.width = hash_param.nhash;
  hash_key1.pitch = hash_key1.width * sizeof(decltype(hash_key1)::Type);
  CUDA_CHECK(cudaMalloc((void**)&hash_key1.data, hash_key1.getSizeInBytes()));

  hash_key2.height = __max_sift;
  hash_key2.width = hash_param.nhash;
  hash_key2.pitch = hash_key2.width * sizeof(decltype(hash_key2)::Type);
  CUDA_CHECK(cudaMalloc((void**)&hash_key2.data, hash_key2.getSizeInBytes()));

  hash_key_bucket1.height = hash_param.nhash;
  hash_key_bucket1.width = hash_param.kNumBins;
  hash_key_bucket1.pitch =
      hash_key_bucket1.width * sizeof(decltype(hash_key_bucket1)::Type);
  CUDA_CHECK(
      cudaMalloc((void**)&hash_key_bucket1.data, hash_key_bucket1.getSizeInBytes()));

  hash_value_bucket1.height = hash_param.nhash;
  hash_value_bucket1.width = descriptors1_wrapper_.height;
  hash_value_bucket1.pitch =
      hash_value_bucket1.width * sizeof(decltype(hash_value_bucket1)::Type);
  CUDA_CHECK(cudaMalloc((void**)&hash_value_bucket1.data,
                        hash_value_bucket1.getSizeInBytes()));

  hash_key_bucket2.height = hash_param.nhash;
  hash_key_bucket2.width = hash_param.kNumBins;
  hash_key_bucket2.pitch =
      hash_key_bucket2.width * sizeof(decltype(hash_key_bucket2)::Type);
  CUDA_CHECK(
      cudaMalloc((void**)&hash_key_bucket2.data, hash_key_bucket2.getSizeInBytes()));

  out_dists.height = descriptors2_wrapper_.height;
  out_dists.width = 2;
  out_dists.pitch = out_dists.width * sizeof(decltype(out_dists)::Type);
  CUDA_CHECK(cudaMalloc((void**)&out_dists.data, out_dists.getSizeInBytes()));

  out_ids.height = descriptors2_wrapper_.height;
  out_ids.width = 2;
  out_ids.pitch = out_ids.width * sizeof(decltype(out_ids)::Type);
  CUDA_CHECK(cudaMalloc((void**)&out_ids.data, out_ids.getSizeInBytes()));

  _num_sift[0] = __max_sift;
  _num_sift[1] = __max_sift;

  return true;
}

void SiftMatchCANN::SetMaxSift(int max_sift) {
  max_sift = ((max_sift + 31) / 32) * 32;
  __max_sift = max_sift;
}

void SiftMatchCANN::SetDescriptors(int index,
                                    int num,
                                    const unsigned char* descriptors,
                                    int id /*= -1*/) {
  if (_initialized == 0) return;
  if (index > 1) index = 1;
  if (index < 0) index = 0;
  need_update_hash_cache[index] = 1;
  _have_loc[index] = 0;

  if (id != -1 && _id_sift[index] != -1 && id == _id_sift[index]) {
    need_update_hash_cache[index] = 0;
    return;
  }
  _id_sift[index] = id;
  if (num > __max_sift) num = __max_sift;
  _num_sift[index] = num;
  if (index==0) {
    CUDA_CHECK(cudaMemcpyAsync(descriptors1_wrapper_.data,
                               descriptors,
                               num * 128,
                               cudaMemcpyHostToDevice));
  }else{

    CUDA_CHECK(cudaMemcpyAsync(descriptors2_wrapper_.data,
                               descriptors,
                               num * 128,
                               cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

void SiftMatchCANN::SetDescriptors(int index,
                                    int num,
                                    const float* descriptors,
                                    int id /*= -1*/) {
  if (_initialized == 0) return;

  if (index > 1) index = 1;
  if (index < 0) index = 0;
  need_update_hash_cache[index] = 1;

  if (id != -1 && id == _id_sift[index]) {
    need_update_hash_cache[index] = 0;
    return;
  }

  if (num > __max_sift) num = __max_sift;

  sift_buffer.resize(num * 128 / 4);
  unsigned char* pub = (unsigned char*)&sift_buffer[0];
  for (int i = 0; i < 128 * num; ++i) {
    pub[i] = int(512 * descriptors[i] + 0.5);
  }
  SetDescriptors(index, num, pub, id);
}

int SiftMatchCANN::GetSiftMatch(int max_match,
                                 uint32_t (*match_buffer)[2],
                                 float distmax,
                                 float ratiomax,
                                 int mbm) {
  static int call_count = 0;
  static std::chrono::nanoseconds total_time(0);
  //  std::cout << "cann match begin" << std::endl;
  if (_initialized == 0) return 0;
  if (_num_sift[0] <= 0 || _num_sift[1] <= 0) return 0;

  auto start_time = std::chrono::high_resolution_clock::now();

  descriptors1_wrapper_.height = _num_sift[0];
  descriptors2_wrapper_.height = _num_sift[1];

  dataset1.data = (int8_t*)descriptors1_wrapper_.data;
  dataset1.Dim = 128;
  dataset1.N = descriptors1_wrapper_.height;

  dataset2.data = (int8_t*)descriptors2_wrapper_.data;
  dataset2.Dim = 128;
  dataset2.N = descriptors2_wrapper_.height;

  hash_key1.height = descriptors1_wrapper_.height;
  hash_key2.height = descriptors2_wrapper_.height;

  
  if (need_update_hash_cache[0]) {
    CUDA_CHECK(cudaMemsetAsync(
      hash_key_bucket1.data, 0, hash_key_bucket1.getSizeInBytes(), nullptr));
    hash_value_bucket1.width = descriptors1_wrapper_.height;
    hash_value_bucket1.pitch =
        hash_value_bucket1.width * sizeof(decltype(hash_value_bucket1)::Type);
    CUDA_CHECK(cudaMemsetAsync(hash_value_bucket1.data,
                              0,
                              hash_value_bucket1.getSizeInBytes(),
                              nullptr));
    generate_hash_key(dataset1, hash_param, hash_key1, hash_key_bucket1);
    build_hash_bucket_indexes_version2(
        dataset1, hash_param, hash_key1, hash_key_bucket1, hash_value_bucket1);
  }

  if (need_update_hash_cache[1]) {
    CUDA_CHECK(cudaMemsetAsync(
      hash_key_bucket2.data, 0, hash_key_bucket2.getSizeInBytes(), nullptr));

    generate_hash_key(dataset2, hash_param, hash_key2, hash_key_bucket2);
  }

  out_dists.height = descriptors2_wrapper_.height;
  out_ids.height = descriptors2_wrapper_.height;
  match_descriptor(dataset1,
                   dataset2,
                   hash_param,
                   hash_key_bucket1,
                   hash_value_bucket1,
                   hash_key2,
                   out_dists,
                   out_ids);

  std::vector<float> h_out_dists(out_dists.height * 2);
  std::vector<int> h_out_ids(out_ids.height * 2);
  CUDA_CHECK(cudaMemcpy(
      h_out_dists.data(), out_dists.data, out_dists.getSizeInBytes(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(h_out_ids.data(), out_ids.data, out_ids.getSizeInBytes(), cudaMemcpyDeviceToHost));

  int nmatch = 0;
  for (int query_idx = 0; query_idx < out_dists.height && nmatch < max_match; query_idx++) {
    auto dist = h_out_dists[query_idx * 2];
    auto dist2 = h_out_dists[query_idx * 2 + 1];
    auto train_idx = h_out_ids[query_idx * 2];
    auto train_idx2 = h_out_ids[query_idx * 2 + 1];
//    std::cout << train_idx <<" " << dist  << " "<< train_idx2 << " " << dist2 << std::endl;
    if (train_idx != -1 && train_idx2 == -1 && dist < distmax) // l1 2 square_norm 0.4*0.4
    {
      CHECK(train_idx >= 0 && train_idx < descriptors1_wrapper_.height) << " train_idx: "<< train_idx << " " << descriptors1_wrapper_.height; 
      CHECK(query_idx < descriptors2_wrapper_.height) << " query_idx: "<< query_idx << " " << descriptors2_wrapper_.height; 
      match_buffer[nmatch][0] = train_idx;
      match_buffer[nmatch][1] = query_idx;
      nmatch++;
    } else if (train_idx != -1 && train_idx2 != -1 && dist< distmax && dist < 0.75 * dist2) {
      CHECK(train_idx >=0 &&train_idx < descriptors1_wrapper_.height && train_idx2 < descriptors1_wrapper_.height) << " train_idx: "<< train_idx << " " << descriptors1_wrapper_.height; 
      CHECK(query_idx < descriptors2_wrapper_.height) << " query_idx: "<< query_idx << " " << descriptors2_wrapper_.height; 
      match_buffer[nmatch][0] = train_idx;
      match_buffer[nmatch][1] = query_idx;
      nmatch++;
    }
  }
   
  // std::string match_result_path = "/home/minxuan/Data/3dgs/dji_test/match_result/" + std::to_string(call_count) +".txt";
  // std::ofstream match_file(match_result_path);

  // match_file << "Total matches: " << nmatch << std::endl;
  // match_file << "Format: query_idx train_idx distance ratio" << std::endl;
  // for(int i =0; i < nmatch; i++)
  // {
  //   match_file << match_buffer[i][0] << " " << match_buffer[i][1]  << std::endl;
  // }

  // match_file.close();
  auto end_time = std::chrono::high_resolution_clock::now();


  ++call_count;
  if (call_count == 1)
  {
    return nmatch;
  }
  total_time += end_time - start_time;



  double avg_time = std::chrono::duration_cast<std::chrono::milliseconds >(total_time).count() / static_cast<double>(call_count-1);
  std::cout << "Function called " << call_count << " times. Average time: " << avg_time << " ms. "<< descriptors1_wrapper_.height << " "<<descriptors2_wrapper_.height << std::endl;


//  std::cout << "cann match end nmatch: "<< nmatch << "out_dist "<< out_dists.height << " distmax: "<< distmax << std::endl;
  return nmatch;

}