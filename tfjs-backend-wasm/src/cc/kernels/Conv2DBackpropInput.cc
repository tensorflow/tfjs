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

#include <cmath>
#include <cstddef>
#include <vector>
#include "src/cc/backend.h"

namespace tfjs {
namespace wasm {
extern "C" {
#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

// Follows the implementation in the TFJS CPU backend.
void Conv2DBackpropInput(
    const size_t dy_id, const size_t filter_id, const size_t batch_size,
    const size_t filter_height, const size_t filter_width,
    const size_t in_height, const size_t in_width, const size_t in_channels,
    const size_t out_height, const size_t out_width, const size_t out_channels,
    const size_t stride_height, const size_t stride_width, const size_t top_pad,
    const size_t left_pad, const size_t flt_s0, const size_t flt_s1,
    const size_t flt_s2, const size_t x_batch_stride, const size_t x_row_stride,
    const size_t x_col_stride, const size_t x_channel_stride,
    const size_t y_batch_stride, const size_t y_row_stride,
    const size_t y_col_stride, const size_t y_channel_stride,
    const size_t out_id) {
  auto& dy_info = backend::get_tensor_info(dy_id);
  auto& filter_info = backend::get_tensor_info(filter_id);
  auto& out_info = backend::get_tensor_info_out(out_id);

  const float* dy_buf = dy_info.f32();
  const float* filter_buf = filter_info.f32();
  float* out_buf_ptr = out_info.f32_write();

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t d1 = 0; d1 < in_channels; ++d1) {
      for (size_t xr = 0; xr < in_height; ++xr) {
        size_t xr_corner = xr - top_pad;
        int stride_height_multiples = ceil(xr_corner / stride_height);
        size_t xr_min = std::max(0, stride_height_multiples);
        size_t yr_max =
            std::min(out_height, (filter_height + xr_corner) / stride_height);

        for (size_t xc = 0; xc < in_width; ++xc) {
          size_t xc_corner = xc - left_pad;
          int stride_width_multiples = ceil(xc_corner / stride_width);
          size_t xC_min = std::max(0, stride_width_multiples);
          size_t yC_max =
              std::min(out_width, (filter_width + xc_corner) / stride_width);

          float dot_prod = 0.0;
          for (size_t yr = xr_min; yr < yr_max; ++yr) {
            size_t wr = yr * stride_height - xr_corner;

            for (size_t yc = xC_min; yc < yC_max; ++yc) {
              size_t wc = yc * stride_width - xc_corner;
              size_t dy_offset =
                  y_batch_stride * b + y_row_stride * yr + y_col_stride * yc;
              size_t flt_offset = flt_s0 * (filter_height - 1 - wr) +
                                  flt_s1 * (filter_width - 1 - wc) +
                                  flt_s2 * d1;

              for (size_t d2 = 0; d2 < out_channels; ++d2) {
                float pixel = dy_buf[dy_offset + y_channel_stride * d2];
                float weight = filter_buf[flt_offset + d2];
                dot_prod += pixel * weight;
              }
            }
          }

          *out_buf_ptr = dot_prod;
          out_buf_ptr++;
        }
      }
    }
  }
}
}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
