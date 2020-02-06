/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_SOFTMAX_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_SOFTMAX_H_

#include "tensorflow/lite/kernels/internal/common.h"
#include <iostream>

namespace tflite {
namespace reference_integer_ops {

template<typename T> struct OutputParams {};
template<> struct OutputParams<int8> {
  static const int output_num_bits = 8;
  static const int32_t output_min = std::numeric_limits<int8>::min();
};
template<> struct OutputParams<int16> {
  static const int output_num_bits = 16 - 1;
  static const int32_t output_min = 0;
  /** static const int32_t output_min = 0; */
};

// Quantized softmax with int8/int16 integer input and output.
template <typename InputT, typename OutputT>
inline void Softmax(const SoftmaxParams& params,
                    const RuntimeShape& input_shape, const InputT* input_data,
                    const RuntimeShape& output_shape, OutputT* output_data) {
  const int32_t input_beta_multiplier = params.input_multiplier;
  const int32_t input_beta_left_shift = params.input_left_shift;
  const int diff_min = params.diff_min;
  // The representation chosen for the input to the exp() function is Q5.26.
  // We need to leave extra space since values that we skip might be as large as
  // -32 before multiplying by input_beta_multiplier, and therefore as large as
  // -16 afterwards.  Note that exp(-8) is definitely not insignificant to
  // accumulation, but exp(-16) definitely is.
  static const int kScaledDiffIntegerBits = 5;
  static const int kAccumulationIntegerBits = 12;
  using FixedPointScaledDiff =
      gemmlowp::FixedPoint<int32_t, kScaledDiffIntegerBits>;
  using FixedPointAccum =
      gemmlowp::FixedPoint<int32_t, kAccumulationIntegerBits>;
  using FixedPoint0 = gemmlowp::FixedPoint<int32_t, 0>;

  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  int32_t in_min_num = std::numeric_limits<InputT>::min();
  int32_t in_max_num = std::numeric_limits<InputT>::max();
  int32_t out_min_num = std::numeric_limits<OutputT>::min();
  int32_t out_max_num = std::numeric_limits<OutputT>::max();
  OutputParams<OutputT> output_params;
  for (int i = 0; i < outer_size; ++i) {
    InputT max_in_row = in_min_num;
    for (int c = 0; c < depth; ++c) {
      max_in_row = std::max(max_in_row, input_data[i * depth + c]);
    }

    FixedPointAccum sum_of_exps = FixedPointAccum::Zero();
    for (int c = 0; c < depth; ++c) {
      int32_t input_diff =
          static_cast<int32_t>(input_data[i * depth + c]) - max_in_row;
      if (input_diff >= diff_min) {
        const int32_t input_diff_rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_beta_multiplier, input_beta_left_shift);
        const FixedPointScaledDiff scaled_diff_f8 =
            FixedPointScaledDiff::FromRaw(input_diff_rescaled);
        sum_of_exps = sum_of_exps + gemmlowp::Rescale<kAccumulationIntegerBits>(
                                        exp_on_negative_values(scaled_diff_f8));
      }
    }

    int num_bits_over_unit;
    FixedPoint0 shifted_scale = FixedPoint0::FromRaw(GetReciprocal(
        sum_of_exps.raw(), kAccumulationIntegerBits, &num_bits_over_unit));

    std::cout << diff_min << " " << input_beta_multiplier << " " << input_beta_left_shift << std::endl;
    for (int c = 0; c < depth; ++c) {
      int32_t input_diff =
          static_cast<int32_t>(input_data[i * depth + c]) - max_in_row;
      if (input_diff >= diff_min) {
        const int32_t input_diff_rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_beta_multiplier, input_beta_left_shift);
        const FixedPointScaledDiff scaled_diff_f8 =
            FixedPointScaledDiff::FromRaw(input_diff_rescaled);

        FixedPoint0 exp_in_0 = exp_on_negative_values(scaled_diff_f8);
        const int32_t unsat_output = gemmlowp::RoundingDivideByPOT(
            (shifted_scale * exp_in_0).raw(), num_bits_over_unit + 31 - output_params.output_num_bits);
        const int32_t shifted_output = unsat_output + output_params.output_min;

        output_data[i * depth + c] =
            static_cast<OutputT>(std::max(std::min(shifted_output, out_max_num), out_min_num));

      } else {
        std::cout << "input_diff too small" << std::endl;
        output_data[i * depth + c] = output_params.output_min;
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_SOFTMAX_H_
