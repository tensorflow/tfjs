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
// - Tensor `x` and `out` must have dtype float32 (checked in tfjs-core)
// - Tensor `x` and `out` must have data format 'NDHWC' (checked in tfjs-core)
void MaxPool3D(int x_id, int out_id, int batch_size, int channel_size,
               int in_depth, int in_height, int in_width, int out_depth,
               int out_height, int out_width, int stride_depth,
               int stride_height, int stride_width, int dilation_depth,
               int dilation_height, int dilation_width,
               int effective_filter_depth, int effective_filter_height,
               int effective_filter_width, int pad_front, int pad_top,
               int pad_left) {
  const TensorInfo& x_info = backend::get_tensor_info(x_id);
  TensorInfo& out_info = backend::get_tensor_info_out(out_id);

  NDHWCPool3DImpl(
      x_info.f32(),
      NDHWCPool3DInfo{
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
      },
      /*filter_init=*/
      []() -> float { return std::numeric_limits<float>::min(); },
      /*filter_apply=*/
      [](float& data, int, const float& val) { data = std::max(data, val); },
      /*filter_assign=*/
      [buf = out_info.f32_write()](int offset, const float& data) {
        buf[offset] = data;
      });
}

}  // extern "C"
}  // namespace tfjs::wasm
