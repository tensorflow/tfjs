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
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/cc/backend.h"
#include "src/cc/kernels/MaxPool.h"
#include "src/cc/util.h"

namespace {
typedef std::array<int, 14> OperatorCacheKey;

std::map<OperatorCacheKey, xnn_operator_t> operator_cache;
}  // namespace

namespace tfjs {
namespace wasm {
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void MaxPool(const int x_id, const int batch_size, const int input_height,
             const int input_width, const int filter_height,
             const int filter_width, int pad_top, int pad_right, int pad_bottom,
             int pad_left, const int dilation_height, const int dilation_width,
             const int stride_height, const int stride_width,
             const int input_channels, const int output_channels,
             const int out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info(out_id);

  const float* x_buf = reinterpret_cast<float*>(x_info.memory_offset);
  float* out_buf = reinterpret_cast<float*>(out_info.memory_offset);

  xnn_operator_t max_pool_op = nullptr;

  const int flags = 0;
  const int channels = input_channels;

  OperatorCacheKey cache_key = {pad_top,         pad_right,     pad_bottom,
                                pad_left,        filter_height, filter_width,
                                stride_height,   stride_width,  dilation_height,
                                dilation_width,  channels,      input_channels,
                                output_channels, flags};

  auto operator_cache_idx = operator_cache.find(cache_key);

  if (operator_cache_idx == operator_cache.end()) {
    float output_min = -std::numeric_limits<float>::infinity();
    float output_max = std::numeric_limits<float>::infinity();

    xnn_status status = xnn_create_max_pooling2d_nhwc_f32(
        pad_top, pad_right, pad_bottom, pad_left, filter_height, filter_width,
        stride_height, stride_width, dilation_height, dilation_width, channels,
        input_channels /* input_pixel_stride */,
        output_channels /* output_pixel_stride */, output_min, output_max,
        flags, &max_pool_op);

    if (status != xnn_status_success) {
      util::warn(
          "XNN status for xnn_create_max_pooling2d_nhwc_f32 is not "
          "successful. ",
          "Got status %d. Use -c dbg to see XNN logs.", status);
    }

    operator_cache.emplace(cache_key, max_pool_op);

    tfjs::backend::xnn_operator_count++;
  } else {
    max_pool_op = operator_cache_idx->second;
  }

  xnn_status status = xnn_setup_max_pooling2d_nhwc_f32(
      max_pool_op, batch_size, input_height, input_width, x_buf, out_buf,
      nullptr /* thread pool */);
  if (status != xnn_status_success) {
    util::warn(
        "XNN status for xnn_setup_max_pooling2d_nhwc_f32 is not successful. "
        "Got status %d. Use -c dbg to see XNN logs.",
        status);
  }

  xnn_run_operator(max_pool_op, nullptr /* thread pool */);
}
}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
