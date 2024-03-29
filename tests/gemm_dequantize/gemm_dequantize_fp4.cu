#include <iostream>
#include <vector>
#include <type_traits>

#include "cutlass/cutlass.h"

#include "src/fastertransformer/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "src/fastertransformer/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "cutlass_extensions/interleaved_numeric_conversion.h"

#include "cutlass/numeric_types.h"

#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>

using namespace torch_ext;
using fastertransformer::check;

template<typename WeightType>
torch::Tensor fused_gemm_dq(const int m, const int n, const int k, const int64_t timing_iterations, float& avg_time)
{
    torch::Tensor input_activations;
    if (m != k) {
        input_activations = torch::randn({m, k}, torch::dtype(torch::kFloat16).device(torch::kCUDA).requires_grad(false));
    }
    else {
        input_activations = torch::eye(m, torch::dtype(torch::kFloat16).device(torch::kCUDA).requires_grad(false));
    }
    // packed fp4
    auto weight = torch::randint(0, 15, {k, n / 2}).to(torch::dtype(torch::kInt8).device(torch::kCPU).requires_grad(false));
    torch::save(weight, "/root/tensors/weight.pt");
    // auto tmp_ptr = get_ptr<char>(weight);
    // for (int i = 0; i < k * n / 2; i++) {
    //     char value = *(tmp_ptr + i);
    //     *(tmp_ptr + i) = ((value & 0x0f) << 4) | (value >> 4);
    // }
    weight = weight.to(torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));
    auto scales = torch::ones({n}, torch::dtype(torch::kFloat16).device(torch::kCUDA).requires_grad(false));
    auto                 stream = at::cuda::getCurrentCUDAStream().stream();

    const half*          input_act_ptr = get_ptr<const half>(input_activations);
    const WeightType*    weight_ptr    = get_ptr<const WeightType>(weight);
    const half*          scales_ptr    = get_ptr<const half>(scales);

    fastertransformer::CutlassFpAIntBGemmRunner<half, WeightType> fused_gemm_dq_runner;
    const int ws_bytes = fused_gemm_dq_runner.getWorkspaceSize(m, n, k);

    auto output_tensor = torch::empty({m, n}, torch::dtype(torch::kFloat16).device(torch::kCUDA).requires_grad(false));
    auto ws_tensor     = torch::empty({ws_bytes}, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));

    half*   output_tensor_ptr = get_ptr<half>(output_tensor);
    char* ws_ptr            = get_ptr<char>(ws_tensor);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    for (int64_t iter = 0; iter < timing_iterations; ++iter) {
        fused_gemm_dq_runner.gemm(
            input_act_ptr, weight_ptr, scales_ptr, output_tensor_ptr, m, n, k, ws_ptr, ws_bytes, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float total_time_ms = 0;
    cudaEventElapsedTime(&total_time_ms, start, stop);
    avg_time = total_time_ms / float(timing_iterations);

    torch::save(input_activations, "/root/tensors/input_activations.pt");
    torch::save(scales, "/root/tensors/scales.pt");
    torch::save(output_tensor, "/root/tensors/output_tensor.pt");

    return output_tensor;
}

template<typename WeightType>
torch::Tensor fused_gemm(const int m, const int n, const int k, const int64_t timing_iterations, float& avg_time)
{
    torch::Tensor input_activations;
    if (m != k) {
        input_activations = torch::randn({m, k}, torch::dtype(torch::kFloat16).device(torch::kCUDA).requires_grad(false));
    }
    else {
        input_activations = torch::eye(m, torch::dtype(torch::kFloat16).device(torch::kCUDA).requires_grad(false));
    }
    auto weight = torch::randn({k, n}, torch::dtype(torch::kFloat16).device(torch::kCUDA).requires_grad(false));
    auto scales = torch::ones({n}, torch::dtype(torch::kFloat16).device(torch::kCUDA).requires_grad(false));
    auto                 stream = at::cuda::getCurrentCUDAStream().stream();
    auto total_rows_before_expert = torch::ones({1}, torch::dtype(torch::kInt64).device(torch::kCUDA).requires_grad(false));
    total_rows_before_expert[0] = m;

    const half*          input_act_ptr = get_ptr<const half>(input_activations);
    const WeightType*    weight_ptr    = get_ptr<const WeightType>(weight);
    const half*          scales_ptr    = get_ptr<const half>(scales);

    fastertransformer::MoeGemmRunner<half, WeightType> gemm_runner;

    auto output_tensor = torch::empty({m, n}, torch::dtype(torch::kFloat16).device(torch::kCUDA).requires_grad(false));

    half*    output_tensor_ptr = get_ptr<half>(output_tensor);
    int64_t* total_rows_before_expert_ptr = get_ptr<int64_t>(total_rows_before_expert);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    for (int64_t iter = 0; iter < timing_iterations; ++iter) {
        gemm_runner.moe_gemm(input_act_ptr, weight_ptr, nullptr, output_tensor_ptr, total_rows_before_expert_ptr, m, n, k, 1, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    check_cuda_error(cudaGetLastError());
    float total_time_ms = 0;
    cudaEventElapsedTime(&total_time_ms, start, stop);
    avg_time = total_time_ms / float(timing_iterations);

    return output_tensor;
}

__global__ void test_quant_dequant()
{
    cutlass::Array<cutlass::half_t, 16> result;
    cutlass::Array<cutlass::fp4_t, 16> source;
    cutlass::FastInterleavedAndBiasedNumericArrayConverter<cutlass::half_t, cutlass::fp4_t, 16> converter_;
    for (int i = 0; i < 8; i++) {
        *(reinterpret_cast<uint8_t*>(&source) + i) = i;
    }
    result = converter_(source);
    for (int i = 0; i < 16; i++) {
        printf("%f\n", float(result[i]));
    }
}

int main()
{
    cudaSetDevice(1);
    float avg_time;
    fused_gemm_dq<cutlass::uint4b_t>(16384, 16384, 16384, 10, avg_time);
    std::cout << "Average Timing: " << avg_time << " ms" << std::endl;
    fused_gemm<half>(16384, 16384, 16384, 10, avg_time);
    std::cout << "Average Timing: " << avg_time << " ms" << std::endl;
    // test_quant_dequant<<<1, 1>>>();
    // cudaDeviceSynchronize();

    return 0;
}
