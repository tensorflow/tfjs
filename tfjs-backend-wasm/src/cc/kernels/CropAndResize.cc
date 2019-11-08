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

// Must match enum in CropAndResize.ts
enum InterpolationMethod {
  BILINEAR = 0,
  NEAREST = 1,
};

namespace {
template <typename T>
void interpolate_bilinear(T* out_buf_ptr, const T* images_buf,
                          std::vector<int> images_strides, int crop_width,
                          int image_width, int image_width_m1, int num_channels,
                          float extrapolation_value, int box_ind, float y_ind,
                          float width_scale, float x1, float x2) {
  float top_ind = floor(y_ind);
  float bottom_ind = ceil(y_ind);
  float y_lerp = y_ind - top_ind;

  for (int x = 0; x < crop_width; ++x) {
    float x_ind = (crop_width > 1) ? x1 * image_width_m1 + x * width_scale
                                   : 0.5 * (x1 + x2) * image_width_m1;

    if (x_ind < 0 || x_ind > image_width - 1) {
      for (int c = 0; c < num_channels; ++c) {
        *out_buf_ptr = extrapolation_value;
        out_buf_ptr++;
      }
      continue;
    }

    float left_ind = floor(x_ind);
    float right_ind = ceil(x_ind);
    float x_lerp = x_ind - left_ind;

    for (int c = 0; c < num_channels; ++c) {
      int ind = c + left_ind * images_strides[2] + top_ind * images_strides[1] +
                box_ind;
      const float top_left = images_buf[ind];

      ind = c + right_ind * images_strides[2] + top_ind * images_strides[1] +
            box_ind;

      const float top_right = images_buf[ind];

      ind = c + left_ind * images_strides[2] + bottom_ind * images_strides[1] +
            box_ind;

      const float bottom_left = images_buf[ind];

      ind = c + right_ind * images_strides[2] + bottom_ind * images_strides[1] +
            box_ind;

      const float bottom_right = images_buf[ind];

      const float top = top_left + (top_right - top_left) * x_lerp;
      const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;

      *out_buf_ptr = top + ((bottom - top) * y_lerp);
      out_buf_ptr++;
    }
  }
}

template <typename T>
void interpolate_nearest(T* out_buf_ptr, const T* images_buf,
                         std::vector<int> images_strides, int crop_width,
                         int image_width, int image_width_m1, int num_channels,
                         float extrapolation_value, int box_ind, float y_ind,
                         float width_scale, float x1, float x2) {
  for (int x = 0; x < crop_width; ++x) {
    const float x_ind = (crop_width > 1) ? x1 * image_width_m1 + x * width_scale
                                         : 0.5 * (x1 + x2) * image_width_m1;

    if (x_ind < 0 || x_ind > image_width - 1) {
      for (int c = 0; c < num_channels; ++c) {
        *out_buf_ptr = extrapolation_value;
        out_buf_ptr++;
      }
      continue;
    }

    float closest_x = round(x_ind);
    float closest_y = round(y_ind);
    for (int c = 0; c < num_channels; ++c) {
      const int in_ind = c + closest_x * images_strides[2] +
                         closest_y * images_strides[1] + box_ind;
      *out_buf_ptr = images_buf[in_ind];
      out_buf_ptr++;
    }
  }
}

}  // namespace

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void CropAndResize(int images_id, int boxes_id, int box_ind_id, int num_boxes,
                   int* images_shape_ptr, int crop_height, int crop_width,
                   InterpolationMethod method, float extrapolation_value,
                   int out_id) {
  const int images_shape_length = 4;
  const std::vector<int>& images_shape = std::vector<int>(
      images_shape_ptr, images_shape_ptr + images_shape_length);
  const auto images_strides = util::compute_strides(images_shape);

  const std::vector<int>& output_shape = {num_boxes, crop_height, crop_width,
                                          images_shape[3]};
  const auto output_strides = util::compute_strides(output_shape);

  auto& images_info = backend::get_tensor_info(images_id);
  auto& boxes_info = backend::get_tensor_info(boxes_id);
  auto& box_ind_info = backend::get_tensor_info(box_ind_id);
  auto& out_info = backend::get_tensor_info_out(out_id);

  const float* images_buf = images_info.f32();
  const int images_size = images_info.size;

  const float* boxes_buf = boxes_info.f32();
  const int boxes_size = boxes_info.size;

  const int* box_ind_buf = box_ind_info.i32();
  const int box_ind_size = box_ind_info.size;

  float* out_buf = out_info.f32_write();
  const int out_size = out_info.size;

  const int batch = images_shape[0];
  const int image_height = images_shape[1];
  const int image_width = images_shape[2];
  const int num_channels = images_shape[3];

  const int image_height_m1 = image_height - 1;
  const int image_width_m1 = image_width - 1;

  for (int b = 0; b < num_boxes; ++b) {
    const float y1 = *boxes_buf;
    boxes_buf++;
    const float x1 = *boxes_buf;
    boxes_buf++;
    const float y2 = *boxes_buf;
    boxes_buf++;
    const float x2 = *boxes_buf;
    boxes_buf++;

    if (*box_ind_buf >= batch) {
      continue;
    }

    const int box_ind = *box_ind_buf * images_strides[0];

    const float height_scale =
        (crop_height > 1) ? (y2 - y1) * image_height_m1 / (crop_height - 1) : 0;
    const float width_scale =
        (crop_width > 1) ? (x2 - x1) * image_width_m1 / (crop_width - 1) : 0;

    const bool crop_size_eq_box_size =
        crop_width == 1 + (x2 - x1) * image_width_m1;
    bool requires_interpolation = false;
    if (method == InterpolationMethod::BILINEAR) {
      const float y_lerp_factor = crop_height > 1
                                      ? y1 * image_height + height_scale
                                      : 0.5 * (y1 + y2) * image_height_m1;

      if (y_lerp_factor - floor(y_lerp_factor) != 0.0) {
        requires_interpolation = true;
      } else {
        const float x_lerp_factor = crop_width > 1
                                        ? x1 * image_width_m1 + width_scale
                                        : 0.5 * (x1 + x2) * image_width_m1;

        if (x_lerp_factor - floor(x_lerp_factor) != 0.0) {
          requires_interpolation = true;
        }
      }
    }

    const bool should_memcpy = x2 > x1 && x1 >= 0 &&
                               crop_size_eq_box_size == true &&
                               requires_interpolation == false;

    for (int y = 0; y < crop_height; ++y) {
      const float y_ind = (crop_height > 1)
                              ? y1 * image_height_m1 + y * height_scale
                              : 0.5 * (y1 + y2) * image_height_m1;

      float* out_buf_ptr =
          out_buf + y * output_strides[1] + b * output_strides[0];

      if (y_ind < 0 || y_ind > image_height - 1) {
        for (int x = 0; x < crop_width; ++x) {
          for (int c = 0; c < num_channels; ++c) {
            *out_buf_ptr = extrapolation_value;
            out_buf_ptr++;
          }
        }
        continue;
      }

      if (should_memcpy) {
        int y_ind_int = y_ind;
        images_buf += (y_ind_int * images_strides[1] + box_ind);

        memcpy(out_buf_ptr, images_buf, sizeof(float) * crop_width);
        continue;
      }

      if (method == InterpolationMethod::BILINEAR) {
        interpolate_bilinear(out_buf_ptr, images_buf, images_strides,
                             crop_width, image_width, image_width_m1,
                             num_channels, extrapolation_value, box_ind, y_ind,
                             width_scale, x1, x2);

      } else {
        interpolate_nearest(out_buf_ptr, images_buf, images_strides, crop_width,
                            image_width, image_width_m1, num_channels,
                            extrapolation_value, box_ind, y_ind, width_scale,
                            x1, x2);
      }
    }

    box_ind_buf++;
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
