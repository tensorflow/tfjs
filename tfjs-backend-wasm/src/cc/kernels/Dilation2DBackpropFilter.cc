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
#include <utility>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/dilation2d_impl.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace tfjs::wasm {

// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

// REQUIRES:
// - Tensor `x_id`, `filter_id`, `dy_id`, and `grad` must have the same dtype.
void Dilation2DBackpropFilter(int x_id, int filter_id, int dy_id, int grad_id,
                              DType dtype, int batch, int depth, int in_height,
                              int in_width, int out_height, int out_width,
                              int stride_height, int stride_width,
                              int dilation_height, int dilation_width,
                              int filter_height, int filter_width, int pad_top,
                              int pad_left) {
  const TensorInfo& x_info = backend::get_tensor_info(x_id);
  const TensorInfo& filter_info = backend::get_tensor_info(filter_id);
  const TensorInfo& dy_info = backend::get_tensor_info(dy_id);
  TensorInfo& grad_info = backend::get_tensor_info_out(grad_id);

  Dilation2DInfo info{
      .batch = batch,
      .depth = depth,
      .in_height = in_height,
      .in_width = in_width,
      .out_height = out_height,
      .out_width = out_width,
      .stride_height = stride_height,
      .stride_width = stride_width,
      .dilation_height = dilation_height,
      .dilation_width = dilation_width,
      .filter_height = filter_height,
      .filter_width = filter_width,
      .pad_top = pad_top,
      .pad_left = pad_left,
  };

  switch (dtype) {
    case DType::float32:
      Dilation2DBackpropFilterImpl(x_info.f32(), filter_info.f32(),
                                   dy_info.f32(), grad_info.f32_write(), info);
      break;
    case DType::int32:
      Dilation2DBackpropFilterImpl(x_info.i32(), filter_info.i32(),
                                   dy_info.i32(), grad_info.i32_write(), info);
      break;
    default:
      util::warn(
          "Dilation2DBackpropFilter for tensor id %d failed. Unsupported dtype "
          "%d",
          x_id, dtype);
      break;
  }
}

}  // extern "C"
}  // namespace tfjs::wasm
