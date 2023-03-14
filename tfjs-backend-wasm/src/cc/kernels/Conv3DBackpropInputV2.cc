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
#include "tfjs-backend-wasm/src/cc/conv3d_impl.h"

namespace tfjs::wasm {

// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

// REQUIRES:
// - Tensor `filter`, `dy`, and `dx` must have dtype float32.
// - Tensor `dy` and `dx` must be in data format 'NDHWC' (checked in tfjs-core).
void Conv3DBackpropInputV2(int filter_id, int dy_id, int dx_id, int batch_size,
                           int in_depth, int in_height, int in_width,
                           int in_channels, int out_depth, int out_height,
                           int out_width, int out_channels, int stride_depth,
                           int stride_height, int stride_width,
                           int dilation_depth, int dilation_height,
                           int dilation_width, int filter_depth,
                           int filter_height, int filter_width, int pad_front,
                           int pad_top, int pad_left) {
  const TensorInfo& filter_info = backend::get_tensor_info(filter_id);
  const TensorInfo& dy_info = backend::get_tensor_info(dy_id);
  TensorInfo& dx_info = backend::get_tensor_info_out(dx_id);

  NDHWCConv3DBackpropInputV2Impl</*IN=*/float, /*FILTER=*/float, /*OUT=*/float>(
      filter_info.f32(), dy_info.f32(), dx_info.f32_write(),
      NDHWCConv3DInfo{
          .batch_size = batch_size,
          .in_depth = in_depth,
          .in_height = in_height,
          .in_width = in_width,
          .in_channels = in_channels,
          .out_depth = out_depth,
          .out_height = out_height,
          .out_width = out_width,
          .out_channels = out_channels,
          .stride_depth = stride_depth,
          .stride_height = stride_height,
          .stride_width = stride_width,
          .dilation_depth = dilation_depth,
          .dilation_height = dilation_height,
          .dilation_width = dilation_width,
          .filter_depth = filter_depth,
          .filter_height = filter_height,
          .filter_width = filter_width,
          .pad_front = pad_front,
          .pad_top = pad_top,
          .pad_left = pad_left,
      });
}

}  // extern "C"
}  // namespace tfjs::wasm
