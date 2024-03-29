#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <future>
#include "cuda_utils.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/utils/allocator.h"
#include "arena.h"
#include "src/fastertransformer/utils/config.h"

namespace fastertransformer {


// there are two things we need to fetch: intermediate weights and output weights
// the workflow:
// ffn_layer calls set_source(layer_2)
// (specially replaced it with set_source(layer_1) for the first time, and do fetching)
// ffn_layer calls sync()             sycn and swap workingspace and dst space
//                                    we start to use dst space now

// ffn_layer calls fetch()            start a new fetching to the working space
// ffn_layer calls set_source(layer_x)
// ffn_layer calls sync()             
// ffn_layer calls fetch()
// ...
template<class ActT, class WeightT = ActT, class BiasT = ActT>
class FetcherContext {
    // TODO: Refactor naming
private:
    WeightT* intermediate_working_       = nullptr;  // GPU
    WeightT* output_working_             = nullptr;  // GPU
    BiasT*   intermediate_bias_working_  = nullptr;  // GPU  
    ActT*    intermediate_scale_working_ = nullptr;  // GPU
    ActT*    output_scale_working_       = nullptr;  // GPU

    WeightT* intermediate_dst_           = nullptr;  // GPU 
    WeightT* output_dst_                 = nullptr;  // GPU 
    BiasT*   intermediate_bias_dst_      = nullptr;  // GPU 
    ActT*    intermediate_scale_dst_     = nullptr;  // GPU 
    ActT*    output_scale_dst_           = nullptr;  // GPU 

    int*  permuted_experts_              = nullptr;  // CPU

    size_t intermediate_w_size_per_expert_;
    size_t output_w_size_per_expert_;
    size_t intermediate_b_size_per_expert_;
    size_t intermediate_scale_size_per_expert_;
    size_t output_scale_size_per_expert_;
    size_t weight_size_per_expert_;

    size_t num_rows_;
    size_t num_experts_;

    bool is_allocate_buffer_;
    IAllocator* allocator_;

    std::string next_layer_name_;
    std::string current_layer_name_;
    std::vector<std::future<void>> futures_;

    const char* next_weight_src_;
    const char* current_weight_src_;

    int num_active_experts_;

public:
    cudaStream_t stream;
    FetchType mode; // 1: FETCH_ON_DEMAND
              // 2: PREFETCH
              // it doesn't affect the functionality, just a signal.

    bool first_time;  // first layer should fetch itself and prefetch next layer
    bool last_time;  //last layer should not prefetch

    // 1. copy to expert_for_source_row_fetching
    // 2. calc expert_sparse_idx_working
    // 3. launch fetch on the stream, from source to working
    void fetch(const int* permuted_experts, bool prefetch);

    // finish previous job
    // drop all previous dst space things and update them.
    void sync(); 

    // called in FfnLayer.cc
    void set_source(const char* next_weight_src, const char* current_weight_src);

    void set_layer(const std::string& next_layer_name, const std::string& current_layer_name, bool is_first_moe, bool is_last_moe);

    std::string get_layer_name() const { return current_layer_name_; }

    void get_weights(int           & num_active_experts,
                     const WeightT*& fc1_expert_weights,
                     const WeightT*& fc2_expert_weights,
                     const BiasT*  & fc1_expert_biases,
                     const ActT*   & fc1_scales,
                     const ActT*   & fc2_scales) const;

    FetcherContext(FetchType mode, 
                   int num_experts, 
                   size_t intermediate_w_size_per_expert,
                   size_t output_w_size_per_expert, 
                   size_t intermediate_b_size_per_expert,
                   size_t intermediate_scale_size_per_expert,
                   size_t output_scale_size_per_expert,
                   size_t arena_size);
    ~FetcherContext();

    void allocateBuffer(IAllocator* allocator, size_t num_rows);
    void freeBuffer();
    
    using tag_t = typename GroupedMemoryArena::tag_t;

    static constexpr bool scales_required =
        std::is_same<WeightT, uint8_t>::value || std::is_same<WeightT, cutlass::uint4b_t>::value ||
        std::is_same<WeightT, cutlass::fp4_t>::value || std::is_same<WeightT, cutlass::nf4_t>::value;

};



} // namespace fastertransformer