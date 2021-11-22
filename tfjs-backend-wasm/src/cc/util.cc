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

}  // namespace util
}  // namespace tfjs
