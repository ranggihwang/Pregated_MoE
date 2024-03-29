#include "src/fastertransformer/utils/arena.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cache_policy.h"
#include "lru_cache_policy.h"
#include "lfu_cache_policy.h"
#include "lifo_cache_policy.h"

namespace fastertransformer {

template class MemoryArena<char>;

template <typename T>
MemoryArena<T>::MemoryArena(size_t size, size_t chunk_size, cudaStream_t stream) 
    : chunk_size_(chunk_size), 
      size_(size),
      chunk_num_(0),
      ptr_(nullptr),
      cache_(nullptr),
      stream_(stream),
      offload_buffer_(nullptr),
      pitch_sizes_(),
      pool_(nullptr)
{
    chunk_num_ = size_ / chunk_size_;

    if (GlobalConfig::instance().cache_policy == "LFU") {
        cache_ = std::make_shared<cache_t>(chunk_num_, std::make_shared<caches::LFUCachePolicy<tag_t>>());
    } else if (GlobalConfig::instance().cache_policy == "LRU") {
        cache_ = std::make_shared<cache_t>(chunk_num_, std::make_shared<caches::LRUCachePolicy<tag_t>>());
    } else if (GlobalConfig::instance().cache_policy == "LIFO") {
        cache_ = std::make_shared<cache_t>(chunk_num_, std::make_shared<caches::LIFOCachePolicy<tag_t>>());
    } else {
        cache_ = std::make_shared<cache_t>(chunk_num_, std::make_shared<caches::NoCachePolicy<tag_t>>());
    }

    // Ensure every experts is aligned
    ptr_ = mallocBuffer(chunk_size_, chunk_num_);
    // Pre-cache
    // This is a workaround, currently this process is necessary
    for (int t = 0; t < chunk_num_; t++) {
        cache_->PutDummy(std::to_string(t), (T*)((char*)ptr_ + pitch_sizes_[chunk_size_] * t));
    }

    if (GlobalConfig::instance().disk_offload) {
        check_cuda_error(cudaMallocHost(&offload_buffer_, chunk_size_ * sizeof(T)));
    }

    pool_ = std::make_shared<ctpl::thread_pool>(1);
}

template <typename T>
std::future<void> MemoryArena<T>::allocate(const tag_t& tag, T* dst, const T* src,
                                           std::function<void(const T*, cudaStream_t)> post_callback)
{
    auto repl = cache_->GetOrPut(tag, nullptr);
    if (GlobalConfig::instance().profiling) {
        Profiling::instance().cacheHit(repl.second);
    }
    auto future = pool_->push([=](int) {
        if (!GlobalConfig::instance().use_cache  // if not use_cache, do this anyway
            || (repl.first != nullptr && !repl.second && src != nullptr)) {
            const T* cpy_src = src;
            if (GlobalConfig::instance().disk_offload) {
                std::string filename = GlobalConfig::instance().offload_path + tag + ".bin";
                std::ifstream ifs(filename, std::ifstream::binary);
                ifs.read(offload_buffer_, chunk_size_ * sizeof(T));
                FT_CHECK_WITH_INFO(ifs, "Read from " + filename + " failed");
                cpy_src = offload_buffer_;
            }
            check_cuda_error(
                cudaMemcpyAsync(
                    repl.first, cpy_src, chunk_size_ * sizeof(T),
                    cudaMemcpyHostToDevice, stream_));
        }
        if (post_callback == nullptr && dst != nullptr) {
            check_cuda_error(cudaMemcpyAsync(dst, repl.first, chunk_size_, cudaMemcpyDeviceToDevice, stream_));
        } else {
            post_callback(repl.first, stream_);
        }
    });
    return future;
}

} // namespace fastertransformer
