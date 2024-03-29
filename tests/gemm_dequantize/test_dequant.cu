#include "cutlass/cutlass.h"
#include "cutlass_extensions/interleaved_numeric_conversion.h"
#include "cutlass/numeric_types.h"

__device__ cutlass::Array<cutlass::half_t, 8> test_quant_dequant();

__device__ cutlass::Array<cutlass::half_t, 8> test_quant_dequant()
{
    cutlass::Array<cutlass::half_t, 8> result;
    cutlass::Array<cutlass::fp4_t, 8> source;
    cutlass::FastInterleavedAndBiasedNumericArrayConverter<cutlass::half_t, cutlass::fp4_t, 8> converter_;
    result = converter_(source);
    return result;
}
