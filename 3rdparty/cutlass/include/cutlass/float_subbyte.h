/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*!
    \file
    \brief Defines a class for using integer types smaller than one byte in host or
      device code.
*/
#pragma once

#if defined(__CUDACC_RTC__)
#include <cuda/std/cstdint>
#else
#include <cstdint>
#endif

#include "cutlass/platform/platform.h"
#include <cuda_fp16.h>
#include "cutlass/bfloat16.h"

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int Bits, bool Normal>
struct quant_map {
  static T const value[];
};

template <typename T>
struct quant_map<T, 4, false> {
  static T const value[];
};

template <typename T>
T const quant_map<T, 4, false>::value[16] = {
  T(0.00000000f),
  T(5.208333333e-03f),
  T(0.66666667f),
  T(1.00000000f),
  T(0.33333333f),
  T(0.50000000f),
  T(0.16666667f),
  T(0.25000000f),
  T(-0.00000000f),
  T(-5.208333333e-03f),
  T(-0.66666667f),
  T(-1.00000000f),
  T(-0.33333333f),
  T(-0.50000000f),
  T(-0.16666667f),
  T(-0.25000000f)
};

template <typename T>
struct quant_map<T, 4, true> {
  static T const value[];
};

template <typename T>
T const quant_map<T, 4, true>::value[16] = {
  T(-1.0f),
  T(-0.6961928009986877f),
  T(-0.5250730514526367f),
  T(-0.39491748809814453f),
  T(-0.28444138169288635f),
  T(-0.18477343022823334f),
  T(-0.09105003625154495f),
  T(0.0f),
  T(0.07958029955625534f),
  T(0.16093020141124725f),
  T(0.24611230194568634f),
  T(0.33791524171829224f),
  T(0.44070982933044434f),
  T(0.5626170039176941f),
  T(0.7229568362236023f),
  T(1.0f)
};

template <typename T, int Bits, bool Normal>
CUTLASS_DEVICE
uint16_t cuda_float_dequantize(uint32_t x)
{
  return 0;
}

template <>
CUTLASS_DEVICE
uint16_t cuda_float_dequantize<half_t, 4, false>(uint32_t x)
{
  static const uint16_t cuda_quant_map[16] = {
    0x0,
    0x1d55,
    0x3955,
    0x3c00,
    0x3555,
    0x3800,
    0x3155,
    0x3400,
    0x8000,
    0x9d55,
    0xb955,
    0xbc00,
    0xb555,
    0xb800,
    0xb155,
    0xb400
  };
  return cuda_quant_map[x];
}

template <>
CUTLASS_DEVICE
uint16_t cuda_float_dequantize<bfloat16_t, 4, false>(uint32_t x)
{
  static const uint16_t cuda_quant_map[16] = {
    0x0,
    0x3bab,
    0x3f2b,
    0x3f80,
    0x3eab,
    0x3f00,
    0x3e2b,
    0x3e80,
    0x8000,
    0xbbab,
    0xbf2b,
    0xbf80,
    0xbeab,
    0xbf00,
    0xbe2b,
    0xbe80
  };
  return cuda_quant_map[x];
}

template <>
CUTLASS_DEVICE
uint16_t cuda_float_dequantize<half_t, 4, true>(uint32_t x)
{
  static const uint16_t cuda_quant_map[16] = {
    0xbc00,
    0xb992,
    0xb833,
    0xb652,
    0xb48d,
    0xb1ea,
    0xadd4,
    0x0,
    0x2d18,
    0x3126,
    0x33e0,
    0x3568,
    0x370d,
    0x3880,
    0x39c9,
    0x3c00
  };
  return cuda_quant_map[x];
}

template <>
CUTLASS_DEVICE
uint16_t cuda_float_dequantize<bfloat16_t, 4, true>(uint32_t x)
{
  static const uint16_t cuda_quant_map[16] = {
    0xbf80,
    0xbf32,
    0xbf06,
    0xbeca,
    0xbe92,
    0xbe3d,
    0xbdba,
    0x0,
    0x3da3,
    0x3e25,
    0x3e7c,
    0x3ead,
    0x3ee2,
    0x3f10,
    0x3f39,
    0x3f80
  };
  return cuda_quant_map[x];
}

template <typename T, int Bits, bool Normal>
CUTLASS_DEVICE
T cuda_float_dequantize_t(uint32_t x)
{
  uint16_t ret_val = cuda_float_dequantize<T, Bits, Normal>(x);
  return reinterpret_cast<T const&>(ret_val);
}

template <typename T, int Bits, bool Normal>
CUTLASS_HOST_DEVICE
T float_dequantize(uint8_t x)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
  return cuda_float_dequantize_t<T, Bits, Normal>(x);
#else
  return quant_map<T, Bits, Normal>::value[x];
#endif
}

template <int Bits, bool Normal>
CUTLASS_HOST_DEVICE
uint8_t float_quantize(float x) { return 0; }

template <>
CUTLASS_HOST_DEVICE
uint8_t float_quantize<4, false>(float x)
{
  // FP4 with bias of 3
  // first bit is a sign
  // subnormals
  // 0b000 = 0
  // 0b001 = 0.0625
  // 0b110 = 2
  // 0b111 = 3
  // 0b100 = 4
  // 0b101 = 6
  // 0b010 = 8
  // 0b011 = 12


  // we do a binary search
  // the pivots are divided by 12 (the FP4 absmax)
  // since we assum input data is in [-1.0, 1.0]

  // !be careful here, its easy to make a mistake
  // that is difficult to noice if you add an extra
  // zero somewhere!

  int sign = x < 0 ? 0b1000 : 0b0000;
  x = fabsf(x);
  if(x > 0.29166667f)
    if( x > 0.583333f)
      if( x > 0.8333333f)
        return 0b0011+sign;
      else
        return 0b0010+sign;
    else
      if(x > 0.4166667f)
        return 0b101+sign;
      else
        return 0b100+sign;
  else
    if(x > 0.0859375f)
      if(x > 0.20833333f)
        return 0b0111+sign;
      else
        return 0b0110+sign;
    else
      if(x > 0.00260417f)
        return 0b0001+sign;
      else
        return 0b0000+sign;
}

template <>
CUTLASS_HOST_DEVICE
uint8_t float_quantize<4, true>(float x)
{

  // the values for this tree was generated by test_normal_map_tree
  // in the file tests/test_functional.py
  if(x > 0.03979014977812767f)
    if(x > 0.3893125355243683f) // 1
      if(x > 0.6427869200706482f) // 11
        if(x > 0.8614784181118011f) // 111
          return 0b1111;
        else
          return 0b1110;
      else
        if(x > 0.5016634166240692f) // 110
          return 0b1101;
        else
          return 0b1100;
    else
      if(x > 0.2035212516784668f) // 10
        if(x > 0.2920137718319893f) // 101
          return 0b1011;
        else
          return 0b1010;
      else
        if(x > 0.1202552504837513f) // 100
          return 0b1001;
        else
          return 0b1000;
  else
    if(x > -0.33967943489551544f) // 0
      if(x > -0.13791173323988914f) // 01
        if(x > -0.045525018125772476f) // 011
          return 0b0111;
        else
          return 0b0110;
      else
        if(x > -0.23460740596055984f) // 010
          return 0b0101;
        else
          return 0b0100;
    else
      if(x > -0.6106329262256622f) // 00
        if(x > -0.4599952697753906f) // 001
          return 0b0011;
        else
          return 0b0010;
      else
        if(x > -0.8480964004993439f) // 000
          return 0b0001;
        else
          return 0b0000;
}

/// 4-bit signed integer type
template <int Bits, bool Normal>
struct float_subbyte {

  /// Number of bits
  static int const kBits = Bits;

  /// Whether float is normal
  static bool const kNormal = Normal;

  /// Storage type
  using Storage = uint8_t;

  /// Bitmask used to truncate from larger integers
  static Storage const kMask = Storage((1 << kBits) - 1);

  //
  // Data members
  //

  Storage storage;

  //
  // Methods
  //

  /// No operation
  CUTLASS_HOST_DEVICE
  float_subbyte() { }

  template <typename S>
  CUTLASS_HOST_DEVICE
  float_subbyte(S value) { 
    storage = float_quantize<Bits, Normal>(static_cast<float>(value));
  }

  CUTLASS_HOST_DEVICE
  explicit operator bfloat16_t() const {
    return float_dequantize<bfloat16_t, Bits, Normal>(storage);
  }

  CUTLASS_HOST_DEVICE
  explicit operator half_t() const {
    return float_dequantize<half_t, Bits, Normal>(storage);
  }

  /// Equality
  CUTLASS_HOST_DEVICE
  bool operator==(float_subbyte const &rhs) const {
    return storage == rhs.storage;
  }

  /// Inequality
  CUTLASS_HOST_DEVICE
  bool operator!=(float_subbyte const &rhs) const {
    return storage != rhs.storage;
  }

  /// Less than or equal
  CUTLASS_HOST_DEVICE
  bool operator<=(float_subbyte const &rhs) const {
    return (*this < rhs) || (*this == rhs);
  }

  /// Greater than or equal
  CUTLASS_HOST_DEVICE
  bool operator>=(float_subbyte const &rhs) const {
    return !(*this < rhs);
  }

  /// Greater than
  CUTLASS_HOST_DEVICE
  bool operator>(float_subbyte const &rhs) const {
    return !(*this <= rhs);
  }
};

// Less than
template <int Bits, bool Normal>
CUTLASS_HOST_DEVICE
bool operator<(float_subbyte<Bits, Normal> const &lhs, float_subbyte<Bits, Normal> const &rhs) {
  return lhs.storage < rhs.storage;
}

// specialization for Normal = false
template <typename T, int Bits>
CUTLASS_HOST_DEVICE
bool operator<(float_subbyte<Bits, false> const &lhs, float_subbyte<Bits, false> const &rhs) {
  return T(lhs) < T(rhs);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// 4-bit 
using fp4_t = float_subbyte<4, false>;

using nf4_t = float_subbyte<4, true>;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines the size of an element in bits - specialized for fp4_t
template <>
struct sizeof_bits<fp4_t> {
  static int const value = 4;
};

/// Defines the size of an element in bits - specialized for nf4_t
template <>
struct sizeof_bits<nf4_t> {
  static int const value = 4;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace platform {

template <>
struct numeric_limits<cutlass::fp4_t> {
  CUTLASS_HOST_DEVICE
  static cutlass::fp4_t const lowest() noexcept { return -1.0f;}
  CUTLASS_HOST_DEVICE
  static cutlass::fp4_t const max() noexcept { return 1.0f;}
  static constexpr bool is_integer = false;
};

template <>
struct numeric_limits<cutlass::nf4_t> {
  CUTLASS_HOST_DEVICE
  static cutlass::nf4_t const lowest() noexcept { return -1.0f;}
  CUTLASS_HOST_DEVICE
  static cutlass::nf4_t const max() noexcept { return 1.0f;}
  static constexpr bool is_integer = false;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace platform
} // namespace cutlass
