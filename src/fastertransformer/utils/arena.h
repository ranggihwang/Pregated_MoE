#pragma once

#include <cuda.h>
#include <unordered_map>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <future>
#include <fstream>
#include <algorithm>
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/config.h"
#include "src/fastertransformer/utils/profiling.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include "src/fastertransformer/utils/ctpl_stl.h"
#include "cache.h"

namespace fastertransformer {

template <typename T>
class MemoryArena {
public:
    using tag_t = std::string;

    MemoryArena(size_t size, size_t chunk_size, cudaStream_t stream);

    MemoryArena(const MemoryArena& o) = delete;

    MemoryArena(MemoryArena&& o)
        : chunk_size_(o.chunk_size_),
          size_(o.size_),
          chunk_num_(o.chunk_num_),
          ptr_(o.ptr_),
          cache_(std::move(o.cache_)),
          stream_(o.stream_),
          offload_buffer_(o.offload_buffer_),
          pitch_sizes_(std::move(o.pitch_sizes_)),
          pool_(std::move(o.pool_))
    {
        o.ptr_ = nullptr;
        o.offload_buffer_ = nullptr;
        o.pool_ = nullptr;
    }

    ~MemoryArena()
    {
        FT_LOG_TRACE("~MemoryArena");
        if (ptr_) {
            cudaFree(ptr_);
        }
        if (offload_buffer_) {
            cudaFreeHost(offload_buffer_);
        }
        FT_LOG_TRACE("~MemoryArena End");
    }

    // Allocate a chunk
    // note: tag < 0 is reserved
    // post_callback is used to do further operations on cached data
    std::future<void> allocate(const tag_t& tag, T* dst = nullptr, const T* src = nullptr, 
                               std::function<void(const T*, cudaStream_t)> post_callback = nullptr);

    // Wait until all previous work is done
    void synchronize()
    {
        check_cuda_error(cudaStreamSynchronize(stream_));
    }

    //
    size_t getCapacity()
    {
        return chunk_num_;
    }

    // Malloc a buffer
    // The buffer is unmanaged and uncached
    T* mallocBuffer(size_t width, size_t height)
    {
        size_t pitch_size;
        T* ptr;
        check_cuda_error(cudaMallocPitch(&ptr, &pitch_size, width * sizeof(T), height));
        pitch_sizes_[width] = pitch_size;
        return ptr;
    }

    size_t getPitchSize(size_t width)
    {
        return pitch_sizes_[width];
    }

    void setStream(cudaStream_t stream)
    {
        stream_ = stream;
    }

private:
    size_t chunk_size_;
    size_t size_;
    size_t chunk_num_;
    T* ptr_;
    using cache_t = caches::fixed_sized_cache<tag_t, T*>;
    std::shared_ptr<cache_t> cache_;
    cudaStream_t stream_;
    char* offload_buffer_;
    std::shared_ptr<ctpl::thread_pool> pool_;
    
    std::unordered_map<size_t, size_t> pitch_sizes_;
};

class GroupedMemoryArena {
public:
    using tag_t = typename MemoryArena<char>::tag_t;

    void init(size_t size, const std::vector<size_t>& tensor_sizes, cudaStream_t stream)
    {
        tensor_sizes_ = tensor_sizes;
        size_t chunk_size = std::accumulate(tensor_sizes_.begin(), tensor_sizes_.end(), 0);
        arena_ = std::make_unique<MemoryArena<char>>(size, chunk_size, stream);
    }

    // reinit if not initialized. the stream will always be reset
    void initIfUninit(size_t size, const std::vector<size_t>& tensor_sizes, cudaStream_t stream)
    {
        if (arena_ == nullptr) {
            init(size, tensor_sizes, stream);
        }
        arena_->setStream(stream);
    }

    std::future<void> allocate(const tag_t& tag, const std::vector<char*>& dsts, const char* src = nullptr)
    {
        FT_CHECK_WITH_INFO(arena_ != nullptr, "Memory arena uninitialized.");
        FT_CHECK(dsts.size() == tensor_sizes_.size());
        auto post_callback = [&tensor_sizes = tensor_sizes_, dsts = dsts](const char* cached_ptr, cudaStream_t stream) {
            const char* ptr = cached_ptr;
            for (int i = 0; i < dsts.size(); ++i) {
                check_cuda_error(cudaMemcpyAsync(dsts[i], ptr, tensor_sizes[i], cudaMemcpyDeviceToDevice, stream));
                ptr += tensor_sizes[i];
            }
        };
        return arena_->allocate(tag, nullptr, src, post_callback);
    }

    char* mallocBuffer(size_t width, size_t height)
    {
        FT_CHECK_WITH_INFO(arena_ != nullptr, "Memory arena uninitialized.");
        return arena_->mallocBuffer(width, height);
    }

    static GroupedMemoryArena& instance()
    {
        static GroupedMemoryArena instance;
        return instance;
    }

private:
    GroupedMemoryArena() {}

    std::unique_ptr<MemoryArena<char>> arena_;

    std::vector<size_t> tensor_sizes_;
};

}  // namespace fastertransformer
