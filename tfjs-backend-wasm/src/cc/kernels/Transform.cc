/* Copyright 2021 Google LLC. All Rights Reserved.
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

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace {
float clamp(const float min, const float x, const float max) {
  return std::max(min, std::min(x, max));
}

float map_coord_constant(const float out_coord, const size_t len) {
  return out_coord;
}

float map_coord_reflect(const float out_coord, const size_t len) {
  // Reflect [abcd] to [dcba|abcd|dcba].
  float in_coord = out_coord;
  if (in_coord < 0) {
    if (len <= 1) {
      in_coord = 0;
    } else {
      size_t sz2 = 2 * len;
      if (in_coord < sz2) {
        in_coord = sz2 * static_cast<int32_t>(-in_coord / sz2) + in_coord;
      }
      in_coord = (in_coord < -len) ? in_coord + sz2 : -in_coord - 1;
    }
  } else if (in_coord > len - 1) {
    if (len <= 1) {
      in_coord = 0;
    } else {
      size_t sz2 = 2 * len;
      in_coord -= sz2 * static_cast<int32_t>(in_coord / sz2);
      if (in_coord >= len) {
        in_coord = sz2 - in_coord - 1;
      }
    }
  }
  // clamp is necessary because when out_coord = 3.5 and len = 4,
  // in_coord = 3.5 and will be rounded to 4 in nearest interpolation.
  return clamp(0, in_coord, len - 1);
}

float map_coord_wrap(const float out_coord, const size_t len) {
  // Wrap [abcd] to [abcd|abcd|abcd].
  float in_coord = out_coord;
  if (in_coord < 0) {
    if (len <= 1) {
      in_coord = 0;
    } else {
      size_t sz = len - 1;
      in_coord += len * (static_cast<int32_t>(-in_coord / sz) + 1);
    }
  } else if (in_coord > len - 1) {
    if (len <= 1) {
      in_coord = 0;
    } else {
      size_t sz = len - 1;
      in_coord -= len * static_cast<int32_t>(in_coord / sz);
    }
  }
  // clamp is necessary because when out_coord = -0.5 and len = 4,
  // in_coord = 3.5 and will be rounded to 4 in nearest interpolation.
  return clamp(0, in_coord, len - 1);
}

float map_coord_nearest(const float out_coord, const size_t len) {
  return clamp(0, out_coord, len - 1);
}

float map_coord(const float out_coord, const size_t len, const size_t mode) {
  switch (mode) {
    case 1:
      return map_coord_constant(out_coord, len);
    case 2:
      return map_coord_reflect(out_coord, len);
    case 3:
      return map_coord_wrap(out_coord, len);
    case 4:
      return map_coord_nearest(out_coord, len);
    default:
      return map_coord_constant(out_coord, len);
  }
}

float read_with_fill_value(const float* image, const size_t image_height,
                           const size_t image_width, const size_t batch_stride,
                           const size_t row_stride, const size_t col_stride,
                           const size_t batch, const int32_t y, const int32_t x,
                           const size_t channel, const float fill_value) {
  // batch and channel must be correct, because they are passed unchanged from
  // the input.
  if (0 <= y && y < image_height && 0 <= x && x < image_width) {
    size_t offset =
        batch * batch_stride + y * row_stride + x * col_stride + channel;
    return image[offset];
  } else {
    return fill_value;
  }
}

float nearest_interpolation(const float* image, const size_t image_height,
                            const size_t image_width, const size_t batch_stride,
                            const size_t row_stride, const size_t col_stride,
                            const size_t batch, const float y, const float x,
                            const size_t channel, const float fill_value) {
  return read_with_fill_value(
      image, image_height, image_width, batch_stride, row_stride, col_stride,
      batch, static_cast<int32_t>(round(y)), static_cast<int32_t>(round(x)),
      channel, fill_value);
}

float bilinear_interpolation(const float* image, const size_t image_height,
                             const size_t image_width,
                             const size_t batch_stride, const size_t row_stride,
                             const size_t col_stride, const size_t batch,
                             const float y, const float x, const size_t channel,
                             const float fill_value) {
  float y_floor = floor(y);
  float x_floor = floor(x);
  float y_ceil = y_floor + 1;
  float x_ceil = x_floor + 1;

  float value_yfloor =
      (x_ceil - x) * read_with_fill_value(image, image_height, image_width,
                                          batch_stride, row_stride, col_stride,
                                          batch, static_cast<int32_t>(y_floor),
                                          static_cast<int32_t>(x_floor),
                                          channel, fill_value) +
      (x - x_floor) * read_with_fill_value(image, image_height, image_width,
                                           batch_stride, row_stride, col_stride,
                                           batch, static_cast<int32_t>(y_floor),
                                           static_cast<int32_t>(x_ceil),
                                           channel, fill_value);

  float value_yceil =
      (x_ceil - x) * read_with_fill_value(image, image_height, image_width,
                                          batch_stride, row_stride, col_stride,
                                          batch, static_cast<int32_t>(y_ceil),
                                          static_cast<int32_t>(x_floor),
                                          channel, fill_value) +
      (x - x_floor) * read_with_fill_value(image, image_height, image_width,
                                           batch_stride, row_stride, col_stride,
                                           batch, static_cast<int32_t>(y_ceil),
                                           static_cast<int32_t>(x_ceil),
                                           channel, fill_value);

  return (y_ceil - y) * value_yfloor + (y - y_floor) * value_yceil;
}
}  // namespace

namespace tfjs {
namespace wasm {
extern "C" {
#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void Transform(const size_t image_id, const size_t transforms_id,
               const bool is_batch_transform, const size_t batch,
               const size_t out_height, const size_t out_width,
               const size_t num_channels, const size_t image_width,
               const size_t image_height, const int32_t* strides_ptr,
               const size_t strides_size, const size_t interpolation_mode_id,
               const size_t fill_mode_id, const float fill_value,
               const size_t out_id) {
  auto& image_info = backend::get_tensor_info(image_id);
  auto& transforms_info = backend::get_tensor_info(transforms_id);
  auto& out_info = backend::get_tensor_info_out(out_id);

  const float* image = image_info.f32();
  const float* transforms = transforms_info.f32();
  float* out_buf = out_info.f32_write();

  const auto image_strides =
      std::vector<size_t>(strides_ptr, strides_ptr + strides_size);
  const size_t batch_stride = image_strides[0];
  const size_t row_stride = image_strides[1];
  const size_t col_stride = image_strides[2];

  // Ref TF implementation:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/image/image_ops.h
  for (size_t b = 0; b < batch; ++b) {
    size_t transforms_offset = 0;
    if (is_batch_transform) {
      transforms_offset = b * 8;
    }

    for (size_t out_y = 0; out_y < out_height; ++out_y) {
      for (size_t out_x = 0; out_x < out_width; ++out_x) {
        for (size_t channel = 0; channel < num_channels; ++channel) {
          float val = 0;
          float projection = transforms[transforms_offset + 6] * out_x +
                             transforms[transforms_offset + 7] * out_y + 1.0;

          if (projection == 0) {
            // Set the fill value for infinite coordinates,
            // which are outside the input image.
            val = fill_value;
          } else {
            float in_x = (transforms[transforms_offset] * out_x +
                          transforms[transforms_offset + 1] * out_y +
                          transforms[transforms_offset + 2]) /
                         projection;
            float in_y = (transforms[transforms_offset + 3] * out_x +
                          transforms[transforms_offset + 4] * out_y +
                          transforms[transforms_offset + 5]) /
                         projection;

            // Map out-of-boundary input coordinates to in-boundary based on
            // fill_mode_id.
            float x = map_coord(in_x, image_width, fill_mode_id);
            float y = map_coord(in_y, image_height, fill_mode_id);

            switch (interpolation_mode_id) {
              case 1:
                val = nearest_interpolation(
                    image, image_height, image_width, batch_stride, row_stride,
                    col_stride, b, y, x, channel, fill_value);
                break;
              case 2:
                val = bilinear_interpolation(
                    image, image_height, image_width, batch_stride, row_stride,
                    col_stride, b, y, x, channel, fill_value);
                break;
              default:
                val = fill_value;
                break;
            }
          }

          size_t offset = b * batch_stride + out_y * row_stride +
                          out_x * col_stride + channel;

          out_buf[offset] = val;
        }
      }
    }
  }
}
}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
