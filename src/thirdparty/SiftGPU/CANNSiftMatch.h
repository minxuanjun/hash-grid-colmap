#pragma once
// std
#include <vector>
// self
#include "SiftGPU.h"
#include "cuda/cann_cuda.h"


class SiftMatchCANN : public SiftMatchGPU {
public:
    explicit SiftMatchCANN(int max_sift);

    virtual ~SiftMatchCANN();

    bool Allocate(int max_sift, int mbm) override;

    void SetMaxSift(int max_sift) override;

    void InitSiftMatch() override;

    void SetDescriptors(int index, int num, const unsigned char *descriptors, int id = -1) override;

    void SetDescriptors(int index, int num, const float *descriptors, int id = -1) override;

    [[deprecated]] void SetFeautreLocation(int index, const float *locations,
                                           int gap) override{}

    [[deprecated]] int GetGuidedSiftMatch(int max_match, uint32_t match_buffer[][2],
                                          float *H, float *F, float distmax,
                                          float ratiomax, float hdistmax,
                                          float fdistmax, int mbm) override{
      return 0;
    }

    int GetSiftMatch(int max_match, uint32_t match_buffer[][2],
                     float distmax, float ratiomax, int mbm) override;

private:
    int GetBestMatch(int max_match, uint32_t match_buffer[][2],
                     float distmax, float ratiomax, int mbm);

private:
    int _num_sift[2];
    int _id_sift[2];
    int need_update_hash_cache[2];
    int _have_loc[2];

    HashParam hash_param;

    DeviceDescriptorWrapper dataset1;
    DeviceDescriptorWrapper dataset2;


    MatrixWrapper<uint8_t> descriptors1_wrapper_;
    MatrixWrapper<uint8_t> descriptors2_wrapper_;
    MatrixWrapper<uint32_t> hash_key1;
    MatrixWrapper<uint32_t> hash_key2;
    MatrixWrapper<uint32_t> hash_key_bucket1;
    MatrixWrapper<uint32_t> hash_key_bucket2;
    MatrixWrapper<uint32_t> hash_value_bucket1;

    MatrixWrapper<float> out_dists;
    MatrixWrapper<int> out_ids;
    //gpu parameter
    int _initialized;
    std::vector<int> sift_buffer;
};
