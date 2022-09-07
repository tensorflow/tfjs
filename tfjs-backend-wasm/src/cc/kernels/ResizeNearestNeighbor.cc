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

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include <cmath>
#include <algorithm>
#include <vector>
#include <cstddef>

#include "tfjs-backend-wasm/src/cc/kernels/ResizeNearestNeighbor.h"

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace tfjs {
namespace wasm {
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void ResizeNearestNeighbor(size_t x_id, size_t batch, size_t old_height,
                    size_t old_width, size_t num_channels, size_t new_height,
                    size_t new_width, bool align_corners,
                    bool half_pixel_centers, size_t out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info_out(out_id);

  const float* x_buf = x_info.f32();
  float* out_buf = out_info.f32_write();

  const std::vector<size_t> x_shape = {
    batch, old_height, old_width, num_channels
  };
  const auto image_strides = util::compute_strides(x_shape);

  const float effective_input_height =
      (align_corners && new_height > 1) ? old_height - 1 : old_height;

  const float effective_input_width =
      (align_corners && new_width > 1) ? old_width - 1 : old_width;


  const float effective_output_height =
      (align_corners && new_height > 1) ? new_height - 1 : new_height;

  const float effective_output_width =
      (align_corners && new_width > 1) ? new_width - 1 : new_width;


  const float height_scale = effective_input_height / effective_output_height;
  const float width_scale = effective_input_width / effective_output_width;

  const float old_height_m1 = old_height - 1;
  const float old_width_m1 = old_width - 1;

  for (int b = 0; b < batch; ++b) {
    const int batch_offset = b * image_strides[0];
    for (int r = 0; r < new_height; ++r) {
      float* out_buf_ptr = out_buf +
        b * (new_height * new_width * num_channels) +
        r * (new_width * num_channels);

      const float y_ind = half_pixel_centers ?

        height_scale * (r + 0.5) :
        height_scale * r;

      int top_ind = std::min(
        old_height_m1,
        align_corners ? std::roundf(y_ind) : std::floor(y_ind));

      if (half_pixel_centers) {
        top_ind = std::max(0, top_ind);
      }

      const float y_lerp = y_ind - top_ind;

      if (width_scale == 1 && y_lerp == 0) {
        memcpy(out_buf_ptr, x_buf + batch_offset + top_ind * image_strides[1],
               sizeof(float) * new_width * num_channels);
      } else {
        for (size_t x = 0; x < new_width; ++x) {
          const float x_ind = half_pixel_centers ?
                            width_scale * (x + 0.5) :
                            width_scale * x;

          int closest_x = std::min(old_width_m1, align_corners ?
                                  std::round(x_ind) :
                                  std::floor(x_ind));

          if (half_pixel_centers) {
            closest_x = std::max(0, closest_x);
          }

          for (size_t c = 0; c < num_channels; ++c) {
            const size_t in_ind = c + closest_x * image_strides[2] +
                                  top_ind * image_strides[1] + batch_offset;
            *out_buf_ptr = x_buf[in_ind];
            out_buf_ptr++;
          }
        }
      }
    }
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
