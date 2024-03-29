# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if (TORCH_VERSION VERSION_GREATER_EQUAL "1.9.0")
    set(gemm_dq_test_files
        th_gemm_dequantize.cc
    )

    add_definitions(-DTORCH_CUDA=1)

    set(LIB_NAME "gemm_dq_unit_ops")
    add_library(${LIB_NAME} SHARED ${gemm_dq_test_files})
    set_target_properties(${LIB_NAME} PROPERTIES
                          CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_link_libraries(${LIB_NAME} "${TORCH_LIBRARIES}" fpA_intB_gemm logger)
else()
    message("TORCH_VERSION ${TORCH_VERSION} < 1.9.0, skipping compiling th_moe_ops.cc because QUInt4x2 is supported after torch 1.9.0")
endif()

set(pybind11_DIR /opt/conda/lib/python3.8/site-packages/pybind11/share/cmake/pybind11)
find_package(pybind11 REQUIRED) 

add_definitions(-DTORCH_CUDA=1)
add_executable(print_dequantize_bytes print_dequantize_bytes.cc)
target_link_libraries(print_dequantize_bytes "${TORCH_LIBRARIES}" fpA_intB_gemm pybind11::embed)

add_definitions(-DTORCH_CUDA=1)
add_executable(gemm_dequantize_fp4 gemm_dequantize_fp4.cu)
target_link_libraries(gemm_dequantize_fp4 "${TORCH_LIBRARIES}" cuda_utils moe_gemm_kernels fpA_intB_gemm logger pybind11::embed)

add_library(test_dequant OBJECT test_dequant.cu)
set_property(TARGET test_dequant PROPERTY CUDA_PTX_COMPILATION ON)
