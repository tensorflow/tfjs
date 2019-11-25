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

#include "src/cc/kernels/FusedDepthwiseConv2D.h"

#include "src/cc/conv2d_impl.h"

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void FusedDepthwiseConv2D(const int x_id, const int batch_size,
                          const int input_height, const int input_width,
                          const int filter_id, const int filter_height,
                          const int filter_width, const int bias_id,
                          int pad_top, int pad_right, int pad_bottom,
                          int pad_left, const int is_same_pad,
                          const int dilation_height, const int dilation_width,
                          const int stride_height, const int stride_width,
                          const int input_channels, const int output_channels,
                          const int activation, const int prelu_weights_id,
                          const int out_id) {
  const bool is_depthwise = true;
  tfjs::wasm::conv2d(x_id, batch_size, input_height, input_width, filter_id,
                     filter_height, filter_width, bias_id, pad_top, pad_right,
                     pad_bottom, pad_left, is_same_pad, dilation_height,
                     dilation_width, stride_height, stride_width,
                     input_channels, output_channels, is_depthwise, activation,
                     prelu_weights_id, out_id);
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
