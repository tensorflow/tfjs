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

#include "src/cc/interpolate_bilinear_impl.h"

#include <cmath>
#include <vector>

#include "src/cc/backend.h"
#include "src/cc/util.h"

namespace tfjs {
namespace wasm {
void interpolate_bilinear(float* out_buf_ptr, const float* images_buf,
                          const std::vector<int> images_strides, int crop_width,
                          int image_width, float image_width_m1,
                          float image_height_m1, int num_channels,
                          float extrapolation_value, int batch_offset,
                          float y_ind, float width_scale, float x1, float x2) {
  int top_ind = std::floor(y_ind);
  int bottom_ind = std::min(image_height_m1, std::ceil(y_ind));
  // int bottom_ind = std::ceil(y_ind);
  float y_lerp = y_ind - top_ind;

  for (int x = 0; x < crop_width; ++x) {
    float x_ind = (crop_width > 1) ? x1 * image_width_m1 + x * width_scale
                                   : 0.5 * (x1 + x2) * image_width_m1;

    if (extrapolation_value != NULL && (x_ind < 0 || x_ind > image_width - 1)) {
      for (int c = 0; c < num_channels; ++c) {
        *out_buf_ptr = extrapolation_value;
        out_buf_ptr++;
      }
      continue;
    }

    int left_ind = std::floor(x_ind);
    int right_ind = std::min(image_width_m1, std::ceil(x_ind));
    // int right_ind = std::ceil(x_ind);
    float x_lerp = x_ind - left_ind;

    for (int c = 0; c < num_channels; ++c) {
      int ind = c + left_ind * images_strides[2] + top_ind * images_strides[1] +
                batch_offset;
      const float top_left = images_buf[ind];

      ind = c + right_ind * images_strides[2] + top_ind * images_strides[1] +
            batch_offset;

      const float top_right = images_buf[ind];

      ind = c + left_ind * images_strides[2] + bottom_ind * images_strides[1] +
            batch_offset;

      const float bottom_left = images_buf[ind];

      ind = c + right_ind * images_strides[2] + bottom_ind * images_strides[1] +
            batch_offset;

      const float bottom_right = images_buf[ind];

      const float top = top_left + (top_right - top_left) * x_lerp;
      const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;

      // util::warn("top_left %f", top_left);
      // util::warn("top_right %f", top_right);
      // util::warn("bottom_left %f", bottom_left);
      // util::warn("bottom_right %f", bottom_right);

      *out_buf_ptr = top + ((bottom - top) * y_lerp);
      out_buf_ptr++;
    }
  }
}

}  // namespace wasm
}  // namespace tfjs
