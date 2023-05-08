/* Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ===========================================================================*/

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

#include "tfjs-backend-wasm/src/cc/util.h"

namespace tfjs {
namespace util {

const std::vector<size_t> compute_strides(const std::vector<size_t> shape) {
  const size_t rank = shape.size();

  if (rank < 2) {
    return {};
  }

  // Last dimension has implicit stride of 1, thus having D-1 (instead of D)
  // strides.
  std::vector<size_t> strides(rank - 1);
  strides[rank - 2] = shape[rank - 1];

  if (rank < 3) {
    return strides;
  }

  // We do i < rank here because i <= 0 is always true for unsigned integers and
  // decrementing will wrap to the max int.
  for (size_t i = rank - 3; i < rank; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  return strides;
}

const std::vector<size_t> assert_and_get_broadcast_shape(
    const std::vector<size_t> shape_a, const std::vector<size_t> shape_b) {
  std::vector<size_t> result = {};
  const size_t l = std::max(shape_a.size(), shape_b.size());
  for (size_t i = 0; i < l; ++i) {
    const int a_idx = static_cast<int32_t>(shape_a.size()) - i - 1;
    const size_t a = (a_idx < 0) ? 1 : shape_a[a_idx];
    const int b_idx = static_cast<int32_t>(shape_b.size()) - i - 1;
    const size_t b = b_idx < 0 ? 1 : shape_b[b_idx];
    if (a == 1) {
      result.push_back(b);
    } else if (b == 1) {
      result.push_back(a);
    } else if (a != b) {
      warn("Operands could not be broadcast together, shape mismatch.");
    } else {
      result.push_back(a);
    }
  }
  std::reverse(result.begin(), result.end());
  return result;
}

const std::vector<size_t> get_broadcast_dims(
    const std::vector<size_t> in_shape, const std::vector<size_t> out_shape) {
  const size_t in_rank = in_shape.size();
  const size_t out_rank = out_shape.size();
  std::vector<size_t> dims = {};
  for (int i = 0; i < in_rank; ++i) {
    const int in_dim = in_rank - 1 - i;
    const int out_dim = out_rank - 1 - i;
    const size_t a = in_shape[in_dim];
    const size_t b = out_dim < 0 ? 1 : out_shape[out_dim];
    if (b > 1 && a == 1) {
      dims.push_back(in_dim);
    }
  }
  std::reverse(dims.begin(), dims.end());
  return dims;
}

const void identity_pool(const size_t x_id, const float* x_buf, float* out_buf,
                         const size_t out_size, const size_t batch_size,
                         const size_t input_height, const size_t input_width,
                         const size_t stride_height, const size_t stride_width,
                         const size_t channels) {
  // Early bailout for the identity case to use memcpy for efficiency.
  if (stride_width == 1 && stride_height == 1) {
    std::memcpy(out_buf, x_buf, out_size * sizeof(*out_buf));
    return;
  }

  // Values per row and column are determined by the stride size.
  // ceil(input_height / stride_height) instead of floor because strides do
  // not guarantee that more than one value is available.
  // e.g. a stride of 3 would 'partition' range(1, 10) into
  // [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
  // and would include 10 in the output: [1, 4, 7, 10]
  size_t vals_per_col = (input_height + stride_height - 1) / stride_height;
  size_t vals_per_row = (input_width + stride_width - 1) / stride_width;

  size_t x_batch_vals_count = input_width * input_height;
  size_t out_batch_vals_count = vals_per_row * vals_per_col;

  // Copy values specified by the strides.
  // Only NHWC is currently supported.
  for (size_t n = 0; n < batch_size; n++) {
    for (size_t h = 0; h < vals_per_col; h++) {
      for (size_t w = 0; w < vals_per_row; w++) {
        for (size_t c = 0; c < channels; c++) {
          size_t x_n_index = n * x_batch_vals_count;
          size_t x_hw_index = h * stride_height * input_width
                              + w * stride_width;
          size_t x_nhw_index = x_n_index + x_hw_index;
          size_t x_nhwc_index = c + channels * x_nhw_index;

          size_t out_n_index = n * out_batch_vals_count;
          size_t out_hw_index = h * vals_per_row + w;
          size_t out_nhw_index = out_n_index + out_hw_index;
          size_t out_nhwc_index = c + channels * out_nhw_index;

          out_buf[out_nhwc_index] = x_buf[x_nhwc_index];
        }
      }
    }
  }
  return;
}

}  // namespace util
}  // namespace tfjs
