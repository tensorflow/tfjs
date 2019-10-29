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

#include <xnnpack.h>
#include <cmath>
#include <limits>
#include <unordered_map>

#include "src/cc/backend.h"
#include "src/cc/util.h"

namespace {
// The operator cache maps the weights id to the xnn_operator_t instantiated for
// // this set of weights.
std::unordered_map<int, xnn_operator_t> operator_cache;

void delete_xnn_operator(int weights_id) {
  xnn_operator_t conv2d_op = operator_cache.at(weights_id);
  xnn_delete_operator(conv2d_op);
  tfjs::backend::xnn_operator_count--;

  operator_cache.erase(weights_id);
}
}  // namespace

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void Conv2D(int x_id, int batch_size, int input_height, int input_width,
            int filter_id, int filter_height, int filter_width, int pad_top,
            int pad_right, int pad_bottom, int pad_left, int dilation_height,
            int dilation_width, int stride_height, int stride_width,
            int input_channels, int output_channels, int out_id) {
  const TensorInfo x_info = backend::get_tensor_info(x_id);
  const TensorInfo filter_info = backend::get_tensor_info(filter_id);
  const TensorInfo out_info = backend::get_tensor_info(out_id);

  const float* x_buf = x_info.buf.f32;
  const float* filter_buf = filter_info.buf.f32;
  float* out_buf = out_info.buf.f32;

  xnn_operator_t conv2d_op = nullptr;

  auto operator_cache_idx = operator_cache.find(filter_id);
  if (operator_cache_idx == operator_cache.end()) {
    float output_min = -std::numeric_limits<float>::infinity();
    float output_max = std::numeric_limits<float>::infinity();

    const int flags = 0;
    const int groups = 1;
    const float* bias_buf = nullptr;
    xnn_status status = xnn_create_convolution2d_nhwc_f32(
        pad_top, pad_right, pad_bottom, pad_left, filter_height, filter_width,
        stride_height, stride_width, dilation_height, dilation_width, groups,
        input_channels, output_channels, input_channels, output_channels,
        filter_buf, bias_buf, -std::numeric_limits<float>::infinity(),
        +std::numeric_limits<float>::infinity(), flags, &conv2d_op);
    if (status != xnn_status_success) {
      util::warn(
          "XNN status for xnn_create_convolution2d_nhwc_f32 is not successful. "
          "Got status %d. Use -c dbg to see XNN logs.",
          status);
    }

    operator_cache.insert({filter_id, conv2d_op});

    backend::register_disposal_callback(filter_id, *delete_xnn_operator);

    tfjs::backend::xnn_operator_count++;
  } else {
    conv2d_op = operator_cache_idx->second;
  }

  xnn_status status = xnn_setup_convolution2d_nhwc_f32(
      conv2d_op, batch_size, input_height, input_width, x_buf, out_buf,
      nullptr /* thread pool */);
  if (status != xnn_status_success) {
    util::warn(
        "XNN status for xnn_setup_convolution2d_nhwc_f32 is not successful. "
        "Got status %d. Use -c dbg to see XNN logs.",
        status);
  }

  xnn_run_operator(conv2d_op, nullptr /* thread pool */);
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
