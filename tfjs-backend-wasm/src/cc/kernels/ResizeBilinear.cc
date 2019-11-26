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

#include <vector>

#include <cmath>
#include "src/cc/backend.h"

#include "src/cc/util.h"

namespace tfjs {
namespace wasm {
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void ResizeBilinear(int x_id, int batch, int old_height, int old_width,
                    int num_channels, int new_height, int new_width,
                    int align_corners, int out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info_out(out_id);

  const float* x_buf = x_info.f32();
  float* out_buf = out_info.f32_write();

  const std::vector<int> x_shape = {batch, old_height, old_width, num_channels};
  const auto x_strides = util::compute_strides(x_shape);

  const float effective_input_height =
      (align_corners > 0 && new_height > 1) ? old_height - 1 : old_height;
  const float effective_input_width =
      (align_corners > 0 && new_width > 1) ? old_width - 1 : old_width;

  const float effective_output_height =
      (align_corners > 0 && new_height > 1) ? new_height - 1 : new_height;
  const float effective_output_width =
      (align_corners > 0 && new_width > 1) ? new_width - 1 : new_width;

  const float effective_row_size_ratio =
      effective_input_height / effective_output_height;
  const float effective_col_size_ratio =
      effective_input_width / effective_output_width;

  float old_height_m1 = old_height - 1;
  float old_width_m1 = old_width - 1;

  for (int b = 0; b < batch; ++b) {
    for (int r = 0; r < new_height; ++r) {
      const float source_frac_row = effective_row_size_ratio * r;
      const int source_row_floor = std::floor(source_frac_row);
      const float row_frac = source_frac_row - source_row_floor;

      const int source_row_ceil =
          std::min(old_height_m1, std::ceil(source_frac_row));
      const int top_row_offset =
          b * x_strides[0] + source_row_floor * x_strides[1];
      const int bot_row_offset =
          b * x_strides[0] + source_row_ceil * x_strides[1];

      if (effective_col_size_ratio == 1 && row_frac == 0) {
        memcpy(out_buf, x_buf + top_row_offset,
               sizeof(float) * new_width * num_channels);
        out_buf += (new_width * num_channels);
      } else {
        for (int c = 0; c < new_width; ++c) {
          const float source_frac_col = effective_col_size_ratio * c;
          const int source_col_floor = std::floor(source_frac_col);
          const float col_frac = source_frac_col - source_col_floor;
          const float source_col_ceil =
              std::min(old_width_m1, std::ceil(source_frac_col));

          const int top_left_offset =
              top_row_offset + source_col_floor * x_strides[2];
          const int bot_left_offset =
              bot_row_offset + source_col_floor * x_strides[2];
          const int top_right_offset =
              top_row_offset + source_col_ceil * x_strides[2];
          const int bot_right_offset =
              bot_row_offset + source_col_ceil * x_strides[2];

          const float* x_buf_top_left = x_buf + top_left_offset;
          const float* x_buf_bottom_left = x_buf + bot_left_offset;
          const float* x_buf_top_right = x_buf + top_right_offset;
          const float* x_buf_bottom_right = x_buf + bot_right_offset;

          for (int d = 0; d < num_channels; ++d) {
            const float top_left = *x_buf_top_left;
            const float bottom_left = *x_buf_bottom_left;
            const float top_right = *x_buf_top_right;
            const float bottom_right = *x_buf_bottom_right;

            x_buf_top_left++;
            x_buf_bottom_left++;
            x_buf_top_right++;
            x_buf_bottom_right++;

            const float top = top_left + (top_right - top_left) * col_frac;
            const float bottom =
                bottom_left + (bottom_right - bottom_left) * col_frac;
            const float new_value = top + (bottom - top) * row_frac;
            *out_buf = new_value;
            out_buf++;
          }
        }
      }
    }
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
