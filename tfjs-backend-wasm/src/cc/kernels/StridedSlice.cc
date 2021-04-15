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

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace tfjs {
namespace wasm {
extern "C" {
#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void StridedSlice(const size_t x_id, const int32_t* x_strides_ptr,
                  const size_t x_rank, const int32_t* begin_ptr,
                  const int32_t* end_ptr, const int32_t* strides_ptr,
                  const int32_t* out_shape_ptr, const int32_t* out_strides_ptr,
                  const size_t out_shape_size, const size_t out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  const size_t out_size = out_info.size;

  const float* x_ptr = x_info.f32();
  float* out_buf_ptr = out_info.f32_write();

  const auto x_strides =
      std::vector<size_t>(x_strides_ptr, x_strides_ptr + x_rank - 1);
  const auto begin = std::vector<size_t>(begin_ptr, begin_ptr + x_rank);
  const auto end = std::vector<size_t>(end_ptr, end_ptr + x_rank);
  const auto strides = std::vector<size_t>(strides_ptr, strides_ptr + x_rank);
  const auto out_shape =
      std::vector<size_t>(out_shape_ptr, out_shape_ptr + out_shape_size);
  const auto out_strides = std::vector<size_t>(
      out_strides_ptr, out_strides_ptr + out_shape_size - 1);

  for (size_t i = 0; i < out_size; ++i) {
    auto coords = tfjs::util::offset_to_loc(i, out_strides);

    std::vector<size_t> new_loc = {};
    for (size_t j = 0; j < out_shape_size; ++j) {
      new_loc.push_back(coords[j] * strides[j] + begin[j]);
    }

    const size_t x_index = tfjs::util::loc_to_offset(new_loc, x_strides);
    *out_buf_ptr = x_ptr[x_index];

    out_buf_ptr++;
  }
}
}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
