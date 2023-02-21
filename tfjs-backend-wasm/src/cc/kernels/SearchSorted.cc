/* Copyright 2023 Google LLC.
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

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace tfjs::wasm {

namespace {

template <typename T>
inline void SearchSortedImpl(const T* sorted_sequence_buf, const T* values_buf,
                             int32_t* out_buf, int batch_size,
                             int sequence_size, int values_size,
                             const bool is_lower_bound) {
  for (int b = 0; b < batch_size; ++b) {
    const T* seq_it = sorted_sequence_buf + b * sequence_size;
    const T* val_it = values_buf + b * values_size;
    int32_t* out_it = out_buf + b * values_size;
    for (int i = 0; i < values_size; ++i) {
      if (is_lower_bound) {
        out_it[i] =
            std::lower_bound(seq_it, seq_it + sequence_size, val_it[i]) -
            seq_it;
      } else {
        out_it[i] =
            std::upper_bound(seq_it, seq_it + sequence_size, val_it[i]) -
            seq_it;
      }
    }
  }
}

}  // namespace

// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

// REQUIRES:
// - Tensor `sorted_sequence` must have shape [batch_size, sequence_size]
// (checked in tfjs-core)
// - Tensor `values` must have shape [batch_size, values_size], with batch_size
// equals to sorted_sequence's batch_size (checked in tfjs-core)
// - Tensor `sorted_sequence` and `values` must have the same dtype, as param
// `dtype`.
void SearchSorted(const int sorted_sequence_id, int values_id,
                  const int batch_size, const int sequence_size,
                  const int values_size, const DType dtype,
                  const bool is_side_left, const int out_id) {
  const TensorInfo& sorted_sequence_info =
      backend::get_tensor_info(sorted_sequence_id);
  const TensorInfo& values_info = backend::get_tensor_info(values_id);
  TensorInfo& out_info = backend::get_tensor_info_out(out_id);

  switch (dtype) {
    case DType::float32:
      SearchSortedImpl<float>(sorted_sequence_info.f32(), values_info.f32(),
                              out_info.i32_write(),
                              /*batch_size=*/batch_size,
                              /*sequence_size*/ sequence_size,
                              /*values_size=*/values_size,
                              /*is_lower_bound=*/is_side_left);
      break;
    case DType::int32:
      SearchSortedImpl<int32_t>(sorted_sequence_info.i32(), values_info.i32(),
                                out_info.i32_write(),
                                /*batch_size=*/batch_size,
                                /*sequence_size*/ sequence_size,
                                /*values_size=*/values_size,
                                /*is_lower_bound=*/is_side_left);
      break;
    default:
      util::warn("SearchSorted for tensor id %d failed. Unsupported dtype %d",
                 sorted_sequence_id, dtype);
  }
}

}  // extern "C"
}  // namespace tfjs::wasm
