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
#include "src/cc/util.h"

namespace tfjs {
namespace wasm {
extern "C" {
#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

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
      for (size_t xR = 0; xR < in_height; ++xR) {
        size_t xRCorner = xR - top_pad;
        int stride_height_multiples = ceil(xRCorner / stride_height);
        size_t xRMin = std::max(0, stride_height_multiples);
        size_t yRMax =
            std::min(out_height, (filter_height + xRCorner) / stride_height);

        for (size_t xC = 0; xC < in_width; ++xC) {
          size_t xCCorner = xC - left_pad;
          int stride_width_multiples = ceil(xCCorner / stride_width);
          size_t xCMin = std::max(0, stride_width_multiples);
          size_t yCMax =
              std::min(out_width, (filter_width + xCCorner) / stride_width);

          float dotProd = 0.0;
          for (size_t yR = xRMin; yR < yRMax; ++yR) {
            size_t wR = yR * stride_height - xRCorner;

            for (size_t yC = xCMin; yC < yCMax; ++yC) {
              size_t wC = yC * stride_width - xCCorner;
              size_t dyOffset =
                  y_batch_stride * b + y_row_stride * yR + y_col_stride * yC;
              size_t fltOffset = flt_s0 * (filter_height - 1 - wR) +
                                 flt_s1 * (filter_width - 1 - wC) + flt_s2 * d1;

              for (size_t d2 = 0; d2 < out_channels; ++d2) {
                float pixel = dy_buf[dyOffset + y_channel_stride * d2];
                float weight = filter_buf[fltOffset + d2];

                dotProd += pixel * weight;
              }
            }
          }

          size_t dxOffset = x_batch_stride * b + x_row_stride * xR +
                            x_col_stride * xC + x_channel_stride * d1;
          *out_buf_ptr = dotProd;
          out_buf_ptr++;
        }
      }
    }
  }
}
}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
