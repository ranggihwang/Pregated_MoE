#include "fetcher.h"


#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <sstream>
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/profiling.h"
#include "src/fastertransformer/utils/random.h"
#include <chrono>


namespace fastertransformer {

// the linker asks me to do so

template class FetcherContext<float>;
template class FetcherContext<half>;

template class FetcherContext<float, cutlass::fp4_t>;
template class FetcherContext<float, cutlass::nf4_t>;
template class FetcherContext<float, cutlass::uint4b_t>;
template class FetcherContext<float, cutlass::int4b_t>;

template class FetcherContext<half, cutlass::fp4_t>;
template class FetcherContext<half, cutlass::nf4_t>;
template class FetcherContext<half, cutlass::uint4b_t>;
template class FetcherContext<half, cutlass::int4b_t>;

template class FetcherContext<float, uint8_t>;
template class FetcherContext<half, uint8_t>;

#ifdef ENABLE_BF16
template class FetcherContext<__nv_bfloat16>;

template class FetcherContext<float, __nv_bfloat16>;
template class FetcherContext<half, __nv_bfloat16>;

template class FetcherContext<__nv_bfloat16, float>;
template class FetcherContext<__nv_bfloat16, half>;

template class FetcherContext<__nv_bfloat16, cutlass::fp4_t>;
template class FetcherContext<__nv_bfloat16, cutlass::nf4_t>;
template class FetcherContext<__nv_bfloat16, cutlass::uint4b_t>;
template class FetcherContext<__nv_bfloat16, cutlass::int4b_t>;

template class FetcherContext<__nv_bfloat16, uint8_t>;
#endif


int64_t calc_sparse_time = 0; // microseconds
int64_t cpy_expert_array_to_cpu_time = 0;
int64_t total_row_cpy = 0;
int64_t layer_1_fetch_time = 0;

// 1. copy to expert_for_source_row_fetching
// 2. calc expert_sparse_idx_working
// 3. launch fetch on the stream, from source to working
template<class ActT, class WeightT, class BiasT> 
void FetcherContext<ActT, WeightT, BiasT>::fetch(const int* permuted_experts, bool prefetch)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (last_time && prefetch) {
        FT_LOG_TRACE("Abandon prefetching at final layer");
        return;
    }

    check_cuda_error(cudaMemcpy(permuted_experts_,
                                permuted_experts,
                                sizeof(int) * num_rows_,
                                cudaMemcpyDeviceToHost));

    auto new_end = std::unique(permuted_experts_, permuted_experts_ + num_rows_);
    num_active_experts_ = new_end - permuted_experts_;

    if (GlobalConfig::instance().profiling) {
        Profiling::instance().activeExperts(num_active_experts_);
    }

    if (GlobalConfig::instance().profiling) {
        Profiling::instance().insert(stream, EventType::MEM_START);
    }

    bool fetch_all = GlobalConfig::instance().fetch_all;
    int forced_num_experts = GlobalConfig::instance().forced_num_experts;
    num_active_experts_ = forced_num_experts ? forced_num_experts : num_active_experts_;
    int _active_experts_count = fetch_all ? num_experts_ : num_active_experts_;

    static constexpr bool scales_required =
        std::is_same<WeightT, uint8_t>::value || std::is_same<WeightT, cutlass::uint4b_t>::value ||
        std::is_same<WeightT, cutlass::fp4_t>::value || std::is_same<WeightT, cutlass::nf4_t>::value;

    for(int i = 0; i < _active_experts_count; i++) {
        int expert = (forced_num_experts || fetch_all) ? i : permuted_experts_[i];

        const char* fetch_weight_src = prefetch ? next_weight_src_ : current_weight_src_;
        std::string layer_name = prefetch ? next_layer_name_ : current_layer_name_;

        if (scales_required) {
            futures_.push_back(GroupedMemoryArena::instance().allocate(
                layer_name + "expert" + std::to_string(expert), {
                    reinterpret_cast<char*>(intermediate_working_) + i * intermediate_w_size_per_expert_,
                    reinterpret_cast<char*>(output_working_) + i * output_w_size_per_expert_,
                    reinterpret_cast<char*>(intermediate_scale_working_) + i * intermediate_scale_size_per_expert_,
                    reinterpret_cast<char*>(output_scale_working_) + i * output_scale_size_per_expert_},
                fetch_weight_src + expert * weight_size_per_expert_));
        }
        else {
            futures_.push_back(GroupedMemoryArena::instance().allocate(
                layer_name + "expert" + std::to_string(expert), {
                    reinterpret_cast<char*>(intermediate_working_) + i * intermediate_w_size_per_expert_,
                    reinterpret_cast<char*>(output_working_) + i * output_w_size_per_expert_},
                fetch_weight_src + expert * weight_size_per_expert_));
        }
    }
}

int64_t fetcher_sync_wait_time = 0; // microseconds

template<class ActT, class WeightT, class BiasT> 
void FetcherContext<ActT, WeightT, BiasT>::sync()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    for (auto& future : futures_) {
        future.wait();
    }
    if (GlobalConfig::instance().profiling) {
        Profiling::instance().insert(stream, EventType::MEM_END);
    }
    futures_.clear();
    check_cuda_error(cudaStreamSynchronize(stream));

    // update dst from working (swap them)
    std::swap(intermediate_dst_, intermediate_working_);
    std::swap(output_dst_, output_working_);
    std::swap(intermediate_bias_dst_, intermediate_bias_working_);
    std::swap(intermediate_scale_dst_, intermediate_scale_working_);
    std::swap(output_scale_dst_, output_scale_working_);
}

// called in FfnLayer.cc
// 
template<class ActT, class WeightT, class BiasT> 
void FetcherContext<ActT, WeightT, BiasT>::set_source(const char* next_weight_src,
                                                      const char* current_weight_src)
{
    next_weight_src_ = next_weight_src;
    current_weight_src_ = current_weight_src;
}

template<class ActT, class WeightT, class BiasT> 
void FetcherContext<ActT, WeightT, BiasT>::set_layer(
        const std::string& next_layer_name,
        const std::string& current_layer_name,
        bool is_first_moe,
        bool is_last_moe)
{
    next_layer_name_ = next_layer_name;
    current_layer_name_ = current_layer_name;
    first_time = is_first_moe;
    last_time = is_last_moe;
}

template<class ActT, class WeightT, class BiasT> 
void FetcherContext<ActT, WeightT, BiasT>::get_weights(int           & num_active_experts,
                                                       const WeightT*& fc1_expert_weights,
                                                       const WeightT*& fc2_expert_weights,
                                                       const BiasT*  & fc1_expert_biases,
                                                       const ActT*   & fc1_scales,
                                                       const ActT*   & fc2_scales) const
{
    num_active_experts = num_active_experts_;
    fc1_expert_weights = intermediate_dst_;
    fc2_expert_weights = output_dst_;
    fc1_expert_biases = intermediate_bias_dst_;
    if (scales_required) {
        fc1_scales = intermediate_scale_dst_;
        fc2_scales = output_scale_dst_;
    }
}

int64_t expert_for_row_backup_time = 0; // microseconds

template<class ActT, class WeightT, class BiasT> 
FetcherContext<ActT, WeightT, BiasT>::~FetcherContext() {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_LOG_TRACE("futures left: %d", futures_.size());
    freeBuffer();
    check_cuda_error(cudaStreamDestroy(stream));
}

template<class ActT, class WeightT, class BiasT> 
FetcherContext<ActT, WeightT, BiasT>::FetcherContext(FetchType mode,
                                                     int num_experts, 
                                                     size_t intermediate_w_size_per_expert,
                                                     size_t output_w_size_per_expert,
                                                     size_t intermediate_b_size_per_expert,
                                                     size_t intermediate_scale_size_per_expert,
                                                     size_t output_scale_size_per_expert,
                                                     size_t arena_size) :
    mode(mode),
    first_time(true),
    num_experts_(num_experts),
    intermediate_w_size_per_expert_(cutlass::get_real_size<WeightT>(intermediate_w_size_per_expert)),
    output_w_size_per_expert_(cutlass::get_real_size<WeightT>(output_w_size_per_expert)),
    intermediate_b_size_per_expert_(cutlass::get_real_size<BiasT>(intermediate_b_size_per_expert)),
    intermediate_scale_size_per_expert_(cutlass::get_real_size<ActT>(intermediate_scale_size_per_expert)),
    output_scale_size_per_expert_(cutlass::get_real_size<ActT>(output_scale_size_per_expert)),
    is_allocate_buffer_(false)
{
    // create cuda stream
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    check_cuda_error(cudaStreamCreate(&this->stream));
    weight_size_per_expert_ = intermediate_w_size_per_expert_ + output_w_size_per_expert_ + intermediate_scale_size_per_expert_ + output_scale_size_per_expert_;
    if (scales_required) {
        GroupedMemoryArena::instance().initIfUninit(arena_size, {
            intermediate_w_size_per_expert_,
            output_w_size_per_expert_,
            intermediate_scale_size_per_expert_,
            output_scale_size_per_expert_}, stream);
    }
    else {
        GroupedMemoryArena::instance().initIfUninit(arena_size, {
            intermediate_w_size_per_expert_,
            output_w_size_per_expert_}, stream);
    }
    Profiling::instance().reset();
}


template<class ActT, class WeightT, class BiasT> 
void FetcherContext<ActT, WeightT, BiasT>::allocateBuffer(IAllocator* allocator, size_t num_rows)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        return;
    }

    allocator_ = allocator;
    num_rows_ = num_rows;

    // TODO: refactor with reMalloc
    intermediate_dst_ = (WeightT*)allocator_->reMalloc(intermediate_dst_, intermediate_w_size_per_expert_ * num_experts_);
    output_dst_ = (WeightT*)allocator_->reMalloc(output_dst_, output_w_size_per_expert_ * num_experts_);
    intermediate_bias_dst_ = (BiasT*)allocator_->reMalloc(intermediate_bias_dst_, intermediate_b_size_per_expert_ * num_experts_);
    intermediate_working_ = (WeightT*)allocator_->reMalloc(intermediate_working_, intermediate_w_size_per_expert_ * num_experts_);
    output_working_ = (WeightT*)allocator_->reMalloc(output_working_, output_w_size_per_expert_ * num_experts_);
    intermediate_bias_working_ = (BiasT*)allocator_->reMalloc(intermediate_bias_working_, intermediate_b_size_per_expert_ * num_experts_);
    if (scales_required) {
        intermediate_scale_dst_ = (ActT*)allocator_->reMalloc(intermediate_scale_dst_, intermediate_scale_size_per_expert_ * num_experts_);
        output_scale_dst_ = (ActT*)allocator_->reMalloc(output_scale_dst_, output_scale_size_per_expert_ * num_experts_);
        intermediate_scale_working_ = (ActT*)allocator_->reMalloc(intermediate_scale_working_, intermediate_scale_size_per_expert_ * num_experts_);
        output_scale_working_ = (ActT*)allocator_->reMalloc(output_scale_working_, output_scale_size_per_expert_ * num_experts_);
    }

    permuted_experts_ = (int*)allocator_->reMalloc(permuted_experts_, sizeof(int) * num_rows, false, true);

    is_allocate_buffer_ = true;

    if (GlobalConfig::instance().profiling) {
        Profiling::instance().recordMemoryUsage();
    }
}


template<class ActT, class WeightT, class BiasT> 
void FetcherContext<ActT, WeightT, BiasT>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (is_allocate_buffer_) {
        allocator_->free((void**)&intermediate_dst_);
        allocator_->free((void**)&output_dst_);
        allocator_->free((void**)&intermediate_bias_dst_);
        allocator_->free((void**)&intermediate_working_);
        allocator_->free((void**)&output_working_);
        allocator_->free((void**)&intermediate_bias_working_);
        if (scales_required) {
            allocator_->free((void**)&intermediate_scale_dst_);
            allocator_->free((void**)&output_scale_dst_);
            allocator_->free((void**)&intermediate_scale_working_);
            allocator_->free((void**)&output_scale_working_);
        }

        allocator_->free((void**)&permuted_experts_, true);

        is_allocate_buffer_ = false;
    }
}

} // namespace fastertransformer
