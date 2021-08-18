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

#include <xnnpack.h>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <unordered_map>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/kernels/MaxPool.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace {
typedef std::array<size_t, 14> OperatorCacheKey;

std::map<OperatorCacheKey, xnn_operator_t> operator_cache;
}  // namespace

namespace tfjs {
namespace wasm {
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void MaxPool(const size_t x_id, const size_t batch_size,
             const size_t input_height, const size_t input_width,
             const size_t filter_height, const size_t filter_width,
             size_t pad_top, size_t pad_right, size_t pad_bottom,
             size_t pad_left, const size_t dilation_height,
             const size_t dilation_width, const size_t stride_height,
             const size_t stride_width, const size_t input_channels,
             const size_t output_channels, const size_t out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info(out_id);

  const float* x_buf = reinterpret_cast<float*>(x_info.memory_offset);
  float* out_buf = reinterpret_cast<float*>(out_info.memory_offset);

  xnn_operator_t max_pool_op = nullptr;

  const uint32_t flags = 0;
  const size_t channels = input_channels;

  OperatorCacheKey cache_key = {pad_top,         pad_right,     pad_bottom,
                                pad_left,        filter_height, filter_width,
                                stride_height,   stride_width,  dilation_height,
                                dilation_width,  channels,      input_channels,
                                output_channels, flags};

  auto operator_cache_idx = operator_cache.find(cache_key);

  if (operator_cache_idx == operator_cache.end()) {
    const float output_min = -std::numeric_limits<float>::infinity();
    const float output_max = std::numeric_limits<float>::infinity();

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
      tfjs::backend::threadpool);
  if (status != xnn_status_success) {
    util::warn(
        "XNN status for xnn_setup_max_pooling2d_nhwc_f32 is not successful. "
        "Got status %d. Use -c dbg to see XNN logs.",
        status);
    return;
  }

  xnn_run_operator(max_pool_op, tfjs::backend::threadpool);
}
}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
