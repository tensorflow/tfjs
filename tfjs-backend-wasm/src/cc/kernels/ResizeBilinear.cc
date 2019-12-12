/* Copyright 2019 Google Inc. All Rights Reserved.
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

#include <cmath>
#include <cstddef>
#include <vector>

#include "src/cc/backend.h"
#include "src/cc/interpolate_bilinear_impl.h"

#include "src/cc/util.h"

namespace tfjs {
namespace wasm {
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void ResizeBilinear(size_t x_id, size_t batch, size_t old_height,
                    size_t old_width, size_t num_channels, size_t new_height,
                    size_t new_width, size_t align_corners, size_t out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info_out(out_id);

  const float* x_buf = x_info.f32();
  float* out_buf = out_info.f32_write();

  const std::vector<size_t> x_shape = {batch, old_height, old_width,
                                       num_channels};
  const auto image_strides = util::compute_strides(x_shape);

  const float effective_input_height =
      (align_corners > 0 && new_height > 1) ? old_height - 1 : old_height;
  const float effective_input_width =
      (align_corners > 0 && new_width > 1) ? old_width - 1 : old_width;

  const float effective_output_height =
      (align_corners > 0 && new_height > 1) ? new_height - 1 : new_height;
  const float effective_output_width =
      (align_corners > 0 && new_width > 1) ? new_width - 1 : new_width;

  const float height_scale = effective_input_height / effective_output_height;
  const float width_scale = effective_input_width / effective_output_width;

  const bool should_extrapolate = false;

  const float old_height_m1 = old_height - 1;
  const float old_width_m1 = old_width - 1;

  for (size_t b = 0; b < batch; ++b) {
    for (size_t r = 0; r < new_height; ++r) {
      const float y_ind = height_scale * r;

      float* out_buf_ptr = out_buf +
                           b * (new_height * new_width * num_channels) +
                           r * (new_width * num_channels);

      const size_t top_ind = std::floor(y_ind);
      const size_t bottom_ind = std::min(old_height_m1, std::ceil(y_ind));
      const float y_lerp = y_ind - top_ind;

      const size_t batch_offset = b * image_strides[0];

      if (width_scale == 1 && y_lerp == 0) {
        memcpy(out_buf_ptr, x_buf + batch_offset + top_ind * image_strides[1],
               sizeof(float) * new_width * num_channels);
      } else {
        tfjs::wasm::interpolate_bilinear(
            out_buf_ptr, x_buf, image_strides, new_width, old_width,
            old_width_m1, old_height_m1, num_channels, should_extrapolate, 0.0,
            batch_offset, y_ind, width_scale, 0.0, 0.0);
      }
    }
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
