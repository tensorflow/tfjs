/* Copyright 2019 Google LLC. All Rights Reserved.
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

#include <cstddef>

#include "src/cc/kernels/FusedDepthwiseConv2D.h"

#include "src/cc/backend.h"
#include "src/cc/conv2d_impl.h"

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void FusedDepthwiseConv2D(
    const size_t x_id, const size_t batch_size, const size_t input_height,
    const size_t input_width, const size_t filter_id,
    const size_t filter_height, const size_t filter_width, const size_t bias_id,
    size_t pad_top, size_t pad_right, size_t pad_bottom, size_t pad_left,
    const size_t is_same_pad, const size_t dilation_height,
    const size_t dilation_width, const size_t stride_height,
    const size_t stride_width, const size_t input_channels,
    const size_t output_channels, const FusableActivation activation,
    const size_t prelu_weights_id, const size_t out_id) {
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
