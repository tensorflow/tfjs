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

#include <math.h>
#include <stdio.h>
#include "src/cc/backend.h"

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void CropAndResize(int images_id, int boxes_id, int box_ind_id, int num_boxes,
                   int* images_strides_ptr, int images_strides_length,
                   int* output_strides_ptr, int output_strides_length,
                   int* images_shape_ptr, int images_shape_length,
                   int* crop_size_ptr, int crop_size_length, int method,
                   float extrapolation_value, int out_id) {
  const std::vector<int>& images_strides = std::vector<int>(
      images_strides_ptr, images_strides_ptr + images_strides_length);
  const std::vector<int>& output_strides = std::vector<int>(
      output_strides_ptr, output_strides_ptr + output_strides_length);
  const std::vector<int>& images_shape = std::vector<int>(
      images_shape_ptr, images_shape_ptr + images_shape_length);
  const std::vector<int>& crop_size =
      std::vector<int>(crop_size_ptr, crop_size_ptr + crop_size_length);

  auto& images_info = backend::get_tensor_info(images_id);
  auto& boxes_info = backend::get_tensor_info(boxes_id);
  auto& box_ind_info = backend::get_tensor_info(box_ind_id);
  auto& out_info = backend::get_tensor_info(out_id);

  float* images_buf = reinterpret_cast<float*>(images_info.memory_offset);
  int images_size = images_info.size;

  float* boxes_buf = reinterpret_cast<float*>(boxes_info.memory_offset);
  int boxes_size = boxes_info.size;

  int* box_ind_buf = reinterpret_cast<int*>(box_ind_info.memory_offset);
  int box_ind_size = box_ind_info.size;

  float* out_buf = reinterpret_cast<float*>(out_info.memory_offset);
  int out_size = out_info.size;

  int batch = images_shape[0];
  int image_height = images_shape[1];
  int image_width = images_shape[2];
  int num_channels = images_shape[3];

  int crop_height = crop_size[0];
  int crop_width = crop_size[1];

  // printf("%d \n", images_strides[0]);
  // printf("%d \n", images_strides[1]);

  for (int b = 0; b < num_boxes; ++b) {
    int start_ind = b * 4;
    float y1 = boxes_buf[start_ind++];
    float x1 = boxes_buf[start_ind++];
    float y2 = boxes_buf[start_ind++];
    float x2 = boxes_buf[start_ind++];

    int b_ind = box_ind_buf[b];
    if (b_ind >= batch) {
      continue;
    }

    float height_scale =
        (crop_height > 1)
            ? (y2 - y1) * float(image_height - 1) / float(crop_height - 1)
            : 0;
    float width_scale = (crop_width > 1) ? (x2 - x1) * float(image_width - 1) /
                                               float(crop_width - 1)
                                         : 0;

    for (int y = 0; y < crop_height; ++y) {
      float y_ind = (crop_height > 1)
                        ? y1 * float(image_height - 1) + y * height_scale
                        : 0.5 * (y1 + y2) * float(image_height - 1);

      if (y_ind < 0 || y_ind > image_height - 1) {
        for (int x = 0; x < crop_width; ++x) {
          for (int c = 0; c < num_channels; ++c) {
            int ind = c + x * output_strides[2] + y * output_strides[1] +
                      b * output_strides[0];
            out_buf[ind] = extrapolation_value;
          }
        }
        continue;
      }

      if (method == 0) {  // 'bilinear'
        float top_ind = floor(y_ind);
        float bottom_ind = ceil(y_ind);
        float y_lerp = y_ind - top_ind;

        for (int x = 0; x < crop_width; ++x) {
          float x_ind = (crop_width > 1)
                            ? x1 * float(image_width - 1) + x * width_scale
                            : 0.5 * (x1 + x2) * float(image_width - 1);

          if (x_ind < 0 || x_ind > image_width - 1) {
            for (int c = 0; c < num_channels; ++c) {
              int ind = c + x * output_strides[2] + y * output_strides[1] +
                        b * output_strides[0];
              out_buf[ind] = extrapolation_value;
            }
            continue;
          }

          float left_ind = floor(x_ind);
          float right_ind = ceil(x_ind);
          float x_lerp = x_ind - left_ind;

          for (int c = 0; c < num_channels; ++c) {
            int ind = c + left_ind * images_strides[2] +
                      top_ind * images_strides[1] + b_ind * images_strides[0];
            float top_left = images_buf[ind];

            ind = c + right_ind * images_strides[2] +
                  top_ind * images_strides[1] + b_ind * images_strides[0];

            float top_right = images_buf[ind];

            ind = c + left_ind * images_strides[2] +
                  bottom_ind * images_strides[1] + b_ind * images_strides[0];

            float bottom_left = images_buf[ind];

            ind = c + right_ind * images_strides[2] +
                  bottom_ind * images_strides[1] + b_ind * images_strides[0];

            float bottom_right = images_buf[ind];

            float top = top_left + (top_right - top_left) * x_lerp;
            float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
            ind = c + x * output_strides[2] + y * output_strides[1] +
                  b * output_strides[0];

            // printf("----------- \n");
            // printf("%i \n", ind);
            // printf("%f \n", top_ind);

            out_buf[ind] = top + ((bottom - top) * y_lerp);
          }
        }
      } else {
        for (int x = 0; x < crop_width; ++x) {
          float x_ind = (crop_width > 1)
                            ? x1 * (image_width - 1) + x * width_scale
                            : 0.5 * (x1 + x2) * (image_width - 1);

          if (x_ind < 0 || x_ind > image_width - 1) {
            for (int c = 0; c < num_channels; ++c) {
              int ind = c + x * output_strides[2] + y * output_strides[1] +
                        b * output_strides[0];
              out_buf[ind] = extrapolation_value;
            }
            continue;
          }

          float closest_x = round(x_ind);
          float closest_y = round(y_ind);
          for (int c = 0; c < num_channels; ++c) {
            int in_ind = c + closest_x * images_strides[2] +
                         closest_y * images_strides[1] +
                         b_ind * images_strides[0];
            int out_ind = c + x * output_strides[2] + y * output_strides[1] +
                          b * output_strides[0];
            out_buf[out_ind] = images_buf[in_ind];
          }
        }
      }
    }
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
