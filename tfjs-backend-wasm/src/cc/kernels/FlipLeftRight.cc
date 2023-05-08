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

#include <math.h>
#include <cmath>
#include <cstddef>
#include <vector>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace tfjs {
namespace wasm {

extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void FlipLeftRight(const size_t image_id, const size_t batch,
                   const size_t image_height, const size_t image_width,
                   const size_t num_channels, const size_t out_id) {
  auto& image_info = backend::get_tensor_info(image_id);
  auto& out_info = backend::get_tensor_info_out(out_id);

  const float* image_buf = image_info.f32();
  float* out_buf = out_info.f32_write();

  for (size_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
    const size_t batch_offset =
        batch_idx * image_width * image_height * num_channels;
    for (size_t row = 0; row < image_height; ++row) {
      const size_t row_offset = row * (image_width * num_channels);
      for (size_t col = 0; col < image_width; ++col) {
        const size_t col_offset = col * num_channels;

        for (size_t channel = 0; channel < num_channels; ++channel) {
          const size_t coord_x = image_width - col - 1;
          const size_t image_idx =
              batch_offset + row_offset + col_offset + channel;

          float output_value = image_buf[image_idx];
          // If the coordinate position falls within the image boundaries...
          if (coord_x >= 0 && coord_x < image_width) {
            const size_t flipped_col_offset = coord_x * num_channels;
            const size_t rotated_image_idx =
                batch_offset + row_offset + flipped_col_offset + channel;
            output_value = image_buf[rotated_image_idx];
          }

          *out_buf = output_value;
          out_buf++;
        }
      }
    }
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
