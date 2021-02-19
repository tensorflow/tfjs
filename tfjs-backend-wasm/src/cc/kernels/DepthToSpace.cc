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

void DepthToSpace(const size_t x_id, const size_t block_size,
                  const bool channels_last, const int32_t* x_strides_ptr,
                  const size_t x_strides_size, const int32_t* out_shape_ptr,
                  const int32_t* out_strides_ptr, const size_t out_shape_size,
                  const size_t out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  const size_t out_size = out_info.size;

  const float* x_ptr = x_info.f32();
  float* out_buf_ptr = out_info.f32_write();

  const auto x_strides =
      std::vector<size_t>(x_strides_ptr, x_strides_ptr + x_strides_size);
  const auto out_shape =
      std::vector<size_t>(out_shape_ptr, out_shape_ptr + out_shape_size);
  const auto out_strides = std::vector<size_t>(
      out_strides_ptr, out_strides_ptr + out_shape_size - 1);

  for (size_t i = 0; i < out_size; ++i) {
    auto coords = tfjs::util::offset_to_loc(i, out_strides);
    const size_t b = coords[0];
    const size_t h = channels_last ? coords[1] : coords[2];
    const size_t w = channels_last ? coords[2] : coords[3];
    const size_t d = channels_last ? coords[3] : coords[1];
    const size_t out_depth_size = channels_last ? out_shape[3] : out_shape[1];

    const size_t in_h = h / block_size;
    const size_t offset_h = h % block_size;
    const size_t in_w = w / block_size;
    const size_t offset_w = w % block_size;
    const size_t offset_d = (offset_h * block_size + offset_w) * out_depth_size;

    const size_t in_d = d + offset_d;

    size_t x_index =
        channels_last
            ? tfjs::util::loc_to_offset({b, in_h, in_w, in_d}, x_strides)
            : tfjs::util::loc_to_offset({b, in_d, in_h, in_w}, x_strides);
    *out_buf_ptr = x_ptr[x_index];

    out_buf_ptr++;
  }
}
}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
