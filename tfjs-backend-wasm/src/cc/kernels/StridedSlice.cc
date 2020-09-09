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

#include "src/cc/backend.h"
#include "src/cc/util.h"

namespace tfjs {
namespace wasm {
extern "C" {
#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void StridedSlice(const size_t x_id, const int32_t* begin_ptr,
                  const size_t begin_size, const int32_t* end_ptr,
                  const size_t end_size, const int32_t* strides_ptr,
                  const size_t strides_size, const size_t begin_mask,
                  const size_t end_mask, const size_t ellipsis_mask,
                  const size_t new_axis_mask, const size_t shrink_axis_mask,
                  const size_t out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  const size_t out_size = out_info.size;

  const float* x_ptr = x_info.f32();
  float* out_buf_ptr = out_info.f32_write();

  const auto begin = std::vector<size_t>(begin_ptr, begin_ptr + begin_size);
  const auto end = std::vector<size_t>(end_ptr, end_ptr + end_size);
  const auto strides =
      std::vector<size_t>(strides_ptr, strides_ptr + strides_size);

  for (size_t i = 0; i < out_size; ++i) {
    // auto coords = tfjs::util::offset_to_loc(i, out_strides);
    // const size_t b = coords[0];
    // const size_t h = channels_last ? coords[1] : coords[2];
    // const size_t w = channels_last ? coords[2] : coords[3];
    // const size_t d = channels_last ? coords[3] : coords[1];
    // const size_t out_depth_size = channels_last ? out_shape[3] :
    // out_shape[1];

    // const size_t in_h = h / block_size;
    // const size_t offset_h = h % block_size;
    // const size_t in_w = w / block_size;
    // const size_t offset_w = w % block_size;
    // const size_t offset_d = (offset_h * block_size + offset_w) *
    // out_depth_size;

    // const size_t in_d = d + offset_d;

    // size_t x_index =
    //     channels_last
    //         ? tfjs::util::loc_to_offset({b, in_h, in_w, in_d}, x_strides)
    //         : tfjs::util::loc_to_offset({b, in_d, in_h, in_w}, x_strides);

    // *out_buf_ptr = x_ptr[x_index];

    out_buf_ptr++;
  }
}
}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
