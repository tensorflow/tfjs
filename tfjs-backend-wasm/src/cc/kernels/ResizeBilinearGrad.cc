/* Copyright 2023 Google LLC.
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

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/shape.h"

namespace tfjs::wasm {

namespace {

template <typename T, bool align_corners>
inline void ResizeBilinearGradImpl(const T* x_buf, const T* dy_buf, T* dx_buf,
                                   const Shape<int32_t, 4>& x_shape,
                                   const Shape<int32_t, 4>& dy_shape) {
  const auto [batch, x_height, x_width, depth] = x_shape.array();
  const auto [y_batch, y_height, y_width, y_depth] = dy_shape.array();

  // In the backwards pass, we want to find the pixels that were generated
  // for each pixel in the input image the forward pass and add the
  // corresponding coefficient from dy to the gradient (with some
  // interpolation).
  float height_scale =
      static_cast<float>(x_height - (align_corners && y_height > 1)) /
      static_cast<float>(y_height - (align_corners && y_height > 1));
  float width_scale =
      static_cast<float>(x_width - (align_corners && y_width > 1)) /
      static_cast<float>(y_width - (align_corners && y_width > 1));

  std::fill(dx_buf, dx_buf + x_shape.size(), 0);
  for (int b = 0; b < batch; ++b) {
    for (int r = 0; r < y_height; ++r) {
      float dx_r = r * height_scale;
      int top_dx_r = static_cast<int>(dx_r);
      int bottom_dx_r =
          std::min(static_cast<int>(std::ceil(dx_r)), x_height - 1);
      float dx_r_lerp = dx_r - std::floor(dx_r);

      for (int c = 0; c < y_width; ++c) {
        float dx_c = c * width_scale;
        int left_dx_c = static_cast<int>(dx_c);
        int right_dx_c =
            std::min(static_cast<int>(std::ceil(dx_c)), x_width - 1);
        float dx_c_lerp = dx_c - std::floor(dx_c);

        for (int d = 0; d < depth; ++d) {
          float val = static_cast<float>(dy_buf[dy_shape.offset({b, r, c, d})]);
          dx_buf[x_shape.offset({b, top_dx_r, left_dx_c, d})] +=
              val * (1.0 - dx_r_lerp) * (1.0 - dx_c_lerp);
          dx_buf[x_shape.offset({b, top_dx_r, right_dx_c, d})] +=
              val * (1.0 - dx_r_lerp) * dx_c_lerp;

          dx_buf[x_shape.offset({b, bottom_dx_r, left_dx_c, d})] +=
              val * dx_r_lerp * (1.0 - dx_c_lerp);
          dx_buf[x_shape.offset({b, bottom_dx_r, right_dx_c, d})] +=
              val * dx_r_lerp * dx_c_lerp;
        }
      }
    }
  }
}

}  // namespace

extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

// REQUIRES:
// - Tensor `images`, `dy`, and `dx` must be 4-dimensional (checked in
// tfjs-core).
// - Tensor `dy` must have dtype float32 (guaranteed in forward kernel).
// - Tensor `images` and `dx` must have dtype float32 (casted in kernel).
void ResizeBilinearGrad(int32_t images_id, int32_t dy_id, int32_t dx_id,
                        const int32_t* images_shape_ptr,
                        const int32_t* dy_shape_ptr, bool align_corners) {
  // `x` here refers to `images` in the input.
  const TensorInfo& x_info = backend::get_tensor_info(images_id);
  const TensorInfo& dy_info = backend::get_tensor_info(dy_id);
  TensorInfo& dx_info = backend::get_tensor_info_out(dx_id);

  Shape<int32_t, 4> x_shape(images_shape_ptr);
  Shape<int32_t, 4> dy_shape(dy_shape_ptr);

  if (align_corners) {
    ResizeBilinearGradImpl<float, /*align_corners=*/true>(
        x_info.f32(), dy_info.f32(), dx_info.f32_write(), x_shape, dy_shape);
  } else {
    ResizeBilinearGradImpl<float, /*align_corners=*/false>(
        x_info.f32(), dy_info.f32(), dx_info.f32_write(), x_shape, dy_shape);
  }
}

}  // extern "C"

}  // namespace tfjs::wasm
