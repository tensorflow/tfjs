/* Copyright 2020 Google LLC. All Rights Reserved.
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

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/util.h"

using std::swap;
using std::vector;

namespace tfjs {

template <typename T>
struct ValAndInd {
  T value;
  size_t index;
  bool operator<(const ValAndInd& other) const {
    return value == other.value ? index < other.index : value > other.value;
  }
  bool operator==(const ValAndInd& other) const {
    return value == other.value && index == other.index;
  }
};

template <typename T>
T sign(T value) {
  if (value == 0) return 0;
  return value < 0 ? -1 : 1;
}

template <typename T>
void select(ValAndInd<T>* array, int k, int left, int right) {
  while (right > left) {
    // Use select recursively to sample a smaller set of size s
    // the arbitrary constants 600 and 0.5 are used in the original
    // version to minimize execution time.
    if (right - left > 600) {
      const int n = right - left + 1;
      const int i = k - left + 1;
      const auto z = log(n);
      const auto s = 0.5 * exp(2 * z / 3);
      const auto sd = 0.5 * sqrt(z * s * (n - s) / n) * sign(i - n / 2);
      const int newLeft = std::max(left, static_cast<int>(k - i * s / n + sd));
      const int newRight =
          std::min(right, static_cast<int>(k + (n - i) * s / n + sd));
      select(array, k, newLeft, newRight);
    }
    // partition the elements between left and right around t
    auto t = array[k];
    int i = left;
    int j = right;

    swap(array[left], array[k]);

    if (t < array[right]) {
      swap(array[left], array[right]);
    }
    while (i < j) {
      swap(array[i], array[j]);
      i++;
      j--;
      while (array[i] < t) {
        i = i + 1;
      }
      while (t < array[j]) {
        j = j - 1;
      }
    }
    if (array[left] == t) {
      swap(array[left], array[j]);
    } else {
      j = j + 1;
      swap(array[j], array[right]);
    }
    // Adjust left and right towards the boundaries of the subset
    // containing the (k - left + 1)th smallest element.
    if (j <= k) {
      left = j + 1;
    }
    if (k <= j) {
      right = j - 1;
    }
  }
}

// Based on tfjs-core/src/backends/topk_impl.ts
template <typename T>
void topk(const T* x_data, const size_t x_len, const vector<size_t>& x_shape,
          const int k, const bool sorted, T* out_values_data,
          int32_t* out_indices_data) {
  size_t last_dim = x_shape.back();
  size_t batch = x_len / last_dim;
  size_t size = last_dim;

  for (size_t b = 0; b < batch; b++) {
    size_t offset = b * size;
    vector<ValAndInd<T>> val_and_ind;
    val_and_ind.reserve(size);
    for (size_t i = offset; i < offset + size; i++) {
      val_and_ind.push_back({.value = x_data[i], .index = i - offset});
    }

    if (k < size) {
      select(val_and_ind.data(), k, 0, size - 1);
      val_and_ind.resize(k);
    }

    if (sorted) {
      std::sort(val_and_ind.begin(), val_and_ind.end());
    }

    size_t out_offset = b * k;
    for (size_t i = 0; i < k; i++) {
      size_t index = out_offset + i;
      out_values_data[index] = val_and_ind[i].value;
      out_indices_data[index] = static_cast<int32_t>(val_and_ind[i].index);
    }
  }
}

namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void TopK(const size_t x_id, const size_t* x_shape_ptr,
          const size_t x_shape_length, const DType x_dtype, const int k,
          const bool sorted, const size_t out_values_id,
          const size_t out_indices_id) {
  auto x_shape = vector<size_t>(x_shape_ptr, x_shape_ptr + x_shape_length);
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_values_info = backend::get_tensor_info_out(out_values_id);
  auto& out_indices_info = backend::get_tensor_info_out(out_indices_id);
  switch (x_dtype) {
    case DType::float32:
      topk<float>(x_info.f32(), x_info.size, x_shape, k, sorted,
                  out_values_info.f32_write(), out_indices_info.i32_write());
      break;
    case DType::int32:
      topk<int32_t>(x_info.i32(), x_info.size, x_shape, k, sorted,
                    out_values_info.i32_write(), out_indices_info.i32_write());
      break;
    default:
      util::warn("TopK for tensor id %d failed. Unsupported dtype %d", x_id,
                 x_dtype);
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
