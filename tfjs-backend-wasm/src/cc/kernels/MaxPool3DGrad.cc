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

#include <algorithm>
#include <limits>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/pool3d_impl.h"

namespace tfjs::wasm {

// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

// REQUIRES:
// - Tensor `x`, `dx` and `dy` must have dtype float32 (checked in tfjs-core)
// - Tensor `x`, `dx` and `dy` must have data format 'NDHWC' (checked in
// tfjs-core)
void MaxPool3DGrad(int x_id, int dy_id, int dx_id, int batch_size,
                   int channel_size, int in_depth, int in_height, int in_width,
                   int out_depth, int out_height, int out_width,
                   int stride_depth, int stride_height, int stride_width,
                   int dilation_depth, int dilation_height, int dilation_width,
                   int effective_filter_depth, int effective_filter_height,
                   int effective_filter_width, int pad_front, int pad_top,
                   int pad_left) {
  const TensorInfo& x_info = backend::get_tensor_info(x_id);
  const TensorInfo& dy_info = backend::get_tensor_info(dy_id);
  TensorInfo& dx_info = backend::get_tensor_info_out(dx_id);
  NDHWCPool3DInfo pool3d_info{
      .batch_size = batch_size,
      .channel_size = channel_size,
      .in_depth = in_depth,
      .in_height = in_height,
      .in_width = in_width,
      .out_depth = out_depth,
      .out_height = out_height,
      .out_width = out_width,
      .stride_depth = stride_depth,
      .stride_height = stride_height,
      .stride_width = stride_width,
      .dilation_depth = dilation_depth,
      .dilation_height = dilation_height,
      .dilation_width = dilation_width,
      .effective_filter_depth = effective_filter_depth,
      .effective_filter_height = effective_filter_height,
      .effective_filter_width = effective_filter_width,
      .pad_front = pad_front,
      .pad_top = pad_top,
      .pad_left = pad_left,
  };

  int* max_positions = new int[pool3d_info.out_size()];
  NDHWCPool3DImpl</*IN=*/float>(
      x_info.f32(), pool3d_info,
      /*filter_init=*/
      []() -> std::pair<float, int> {
        return {std::numeric_limits<float>::min(), 0};
      },
      /*filter_apply=*/
      [](std::pair<float, int>& data, int x_offset, const float& x_val) {
        if (x_val >= data.first) {
          data = {x_val, x_offset};
        }
      },
      /*filter_assign=*/
      [max_positions](int offset, const std::pair<float, int>& data) {
        max_positions[offset] = data.second;
      });

  NDHWCPool3DGradImpl(
      dy_info.f32(), dx_info.f32_write(), pool3d_info,
      /*pixel_mask=*/
      [&max_positions](int dy_offset, int dx_offset) {
        return static_cast<float>(dx_offset == max_positions[dy_offset]);
      });

  delete[] max_positions;
}

}  // extern "C"
}  // namespace tfjs::wasm
