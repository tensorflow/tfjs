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

#include <math.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace tfjs {

template <typename T>
struct ValAndInd {
  T value;
  int32_t index;
};

// Based on tfjs-core/src/backends/topk_impl.ts
template <typename T>
void topk(const T* x_data, const size_t x_len,
          const std::vector<size_t>& x_shape, const int k, const bool sorted,
          T* out_values_data, int32_t* out_indices_data) {
  int last_dim = x_shape.back();
  int batch = x_len / last_dim;
  int size = last_dim;

  for (int b = 0; b < batch; b++) {
    int offset = b * size;
    std::vector<ValAndInd<T>> val_and_ind;
    for (int i = offset; i < offset + size; i++) {
      val_and_ind.push_back({.value = x_data[i], .index = i - offset});
    }
    std::sort(val_and_ind.begin(), val_and_ind.end(),
              [](const ValAndInd<T>& a, const ValAndInd<T>& b) -> bool {
                return a.value > b.value;
              });
    int out_offset = b * k;
    for (int i = 0; i < k; i++) {
      int index = out_offset + i;
      out_values_data[index] = val_and_ind[i].value;
      out_indices_data[index] = val_and_ind[i].index;
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
  auto x_shape = std::vector<size_t>(x_shape_ptr, x_shape_ptr + x_shape_length);
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
