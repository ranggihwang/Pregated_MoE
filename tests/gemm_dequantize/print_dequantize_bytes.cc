#include <cublas_v2.h>
#include <iostream>
#include <vector>

#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>

#include "src/fastertransformer/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"

#include "cutlass/numeric_types.h"

template <typename T>
void print_in_short(T const* ptr)
{
    auto tmp = reinterpret_cast<uint16_t const*>(ptr);
    for (int i = 0; i < 16; i++) {
        std::cout << std::hex << *(tmp + i) << "," << std::endl;
    }
    std::cout << std::endl;
}

int main()
{
    print_in_short(cutlass::quant_map<cutlass::half_t, 4, false>::value);
    print_in_short(cutlass::quant_map<cutlass::bfloat16_t, 4, false>::value);
    print_in_short(cutlass::quant_map<cutlass::half_t, 4, true>::value);
    print_in_short(cutlass::quant_map<cutlass::bfloat16_t, 4, true>::value);

    return 0;
}
