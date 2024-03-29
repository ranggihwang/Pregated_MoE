/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/config.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include <fstream>

namespace ft = fastertransformer;

namespace torch_ext {

std::vector<size_t> convert_shape(torch::Tensor tensor)
{
    std::vector<size_t> v_shape;
    for (int i = 0; i < tensor.dim(); i++) {
        v_shape.push_back(tensor.size(i));
    }
    return v_shape;
}

template<typename T>
ft::Tensor convert_tensor(torch::Tensor tensor)
{
    ft::MemoryType mtype = tensor.is_cuda() ? ft::MEMORY_GPU : ft::MEMORY_CPU;
    return convert_tensor<T>(tensor, mtype);
}

template ft::Tensor convert_tensor<int8_t>(torch::Tensor tensor);
template ft::Tensor convert_tensor<float>(torch::Tensor tensor);
template ft::Tensor convert_tensor<half>(torch::Tensor tensor);
#ifdef ENABLE_BF16
template ft::Tensor convert_tensor<__nv_bfloat16>(torch::Tensor tensor);
#endif
template ft::Tensor convert_tensor<int>(torch::Tensor tensor);
template ft::Tensor convert_tensor<unsigned long long int>(torch::Tensor tensor);
template ft::Tensor convert_tensor<unsigned int>(torch::Tensor tensor);
template ft::Tensor convert_tensor<bool>(torch::Tensor tensor);

template<typename T>
ft::Tensor convert_tensor(torch::Tensor tensor, ft::MemoryType memory_type)
{
    return ft::Tensor{memory_type, ft::getTensorType<T>(), convert_shape(tensor), get_ptr<T>(tensor)};
}

template ft::Tensor convert_tensor<int8_t>(torch::Tensor tensor, ft::MemoryType memory_type);
template ft::Tensor convert_tensor<float>(torch::Tensor tensor, ft::MemoryType memory_type);
template ft::Tensor convert_tensor<half>(torch::Tensor tensor, ft::MemoryType memory_type);
#ifdef ENABLE_BF16
template ft::Tensor convert_tensor<__nv_bfloat16>(torch::Tensor tensor, ft::MemoryType memory_type);
#endif
template ft::Tensor convert_tensor<int>(torch::Tensor tensor, ft::MemoryType memory_type);
template ft::Tensor convert_tensor<unsigned long long int>(torch::Tensor tensor, ft::MemoryType memory_type);
template ft::Tensor convert_tensor<unsigned int>(torch::Tensor tensor, ft::MemoryType memory_type);
template ft::Tensor convert_tensor<bool>(torch::Tensor tensor, ft::MemoryType memory_type);

size_t sizeBytes(torch::Tensor tensor)
{
    return tensor.numel() * torch::elementSize(torch::typeMetaToScalarType(tensor.dtype()));
}

template<typename T>
torch::Tensor create_tensor_from_bin(const std::vector<size_t>& shape, const std::string& filename, bool gpu)
{
    return create_tensor_from_bin<T>(shape, std::vector<std::string>{filename}, gpu);
}

template<typename T>
torch::Tensor create_tensor_from_bin(const std::vector<size_t>& shape, const std::vector<std::string>& filenames, bool gpu)
{
    TORCH_CHECK(filenames.size() == 1 || shape[0] == filenames.size(), "shape[0] should have the same size with filenames");

    auto dtype = torch::kFloat32;
    ft::FtCudaDataType model_file_type;
    if (std::is_same<T, float>::value) {
        dtype = torch::kFloat32;
        model_file_type = ft::FtCudaDataType::FP32;
    }
    else if (std::is_same<T, half>::value) {
        dtype = torch::kFloat16;
        model_file_type = ft::FtCudaDataType::FP16;
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        dtype = torch::kFloat16;
        model_file_type = ft::FtCudaDataType::BF16;
    }
#endif
    else {
        dtype = torch::kInt8;
        model_file_type = ft::FtCudaDataType::INT8;
    }
    std::vector<int64_t> sizes(shape.begin(), shape.end());
    torch::IntArrayRef size_array(sizes);
    torch::Tensor tensor =
        gpu ?
        torch::ones(size_array, torch::dtype(dtype).device(torch::kCUDA).requires_grad(false)).contiguous() :
        torch::ones(size_array, torch::dtype(dtype).device(torch::kCPU).pinned_memory(true).requires_grad(false)).contiguous();
    T* ptr = get_ptr<T>(tensor);
    std::vector<size_t> _shape =
        filenames.size() == 1 ?
        shape :
        std::vector<size_t>(shape.begin() + 1, shape.end());
    for (const auto& filename : filenames) {
        // TODO: Refactor loadWeightFromBin
        if (gpu) {
            ft::loadWeightFromBin<T>(ptr, _shape, filename, model_file_type);
        } else {
            ft::loadWeightFromBin<T>(ptr, _shape, filename, model_file_type, false);
        }
        ptr += tensor.numel() / filenames.size();
    }
    return tensor;
}

template torch::Tensor create_tensor_from_bin<float>(const std::vector<size_t>& shape, const std::string& filename, bool gpu);
template torch::Tensor create_tensor_from_bin<float>(const std::vector<size_t>& shape, const std::vector<std::string>& filenames, bool gpu);
template torch::Tensor create_tensor_from_bin<half>(const std::vector<size_t>& shape, const std::string& filename, bool gpu);
template torch::Tensor create_tensor_from_bin<half>(const std::vector<size_t>& shape, const std::vector<std::string>& filenames, bool gpu);
template torch::Tensor create_tensor_from_bin<int8_t>(const std::vector<size_t>& shape, const std::string& filename, bool gpu);
template torch::Tensor create_tensor_from_bin<int8_t>(const std::vector<size_t>& shape, const std::vector<std::string>& filenames, bool gpu);
#ifdef ENABLE_BF16
template torch::Tensor create_tensor_from_bin<__nv_bfloat16>(const std::vector<size_t>& shape, const std::string& filename, bool gpu);
template torch::Tensor create_tensor_from_bin<__nv_bfloat16>(const std::vector<size_t>& shape, const std::vector<std::string>& filenames, bool gpu);
#endif

}  // namespace torch_ext
