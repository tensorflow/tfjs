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

#include "src/cc/backend.h"

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void CropAndResize(int images_id, int boxes_id, int box_ind_id, int num_boxes,
                   const std::vector<int>& images_strides,
                   const std::vector<int>& output_strides,
                   const std::vector<int>& images_shape,
                   const std::vector<int>& crop_size, int method,
                   float extrapolation_value, int out_id) {
  const auto images_info = backend::get_tensor_info(images_id);
  const auto boxes_info = backend::get_tensor_info(boxes_id);
  const auto box_ind_info = backend::get_tensor_info(box_ind_id);
  const auto out_info = backend::get_tensor_info(out_id);

  float* images_buf = images_info.buf.f32;
  int images_size = images_info.size;

  float* boxes_buf = boxes_info.buf.f32;
  int boxes_size = boxes_info.size;

  float* box_ind_buf = box_ind_info.buf.f32;
  int box_ind_size = box_ind_info.size;

  float* out_buf = out_info.buf.f32;
  int out_size = out_info.size;

  for (int b = 0; b < num_boxes; ++b) {
    int startInd = b * 4;
    int y1 = boxes_buf[startInd];
    int x1 = boxes_buf[startInd + 1];
    int y2 = boxes_buf[startInd + 2];
    int x2 = boxes_buf[startInd + 3];
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
