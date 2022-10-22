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

#include <cstring>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif
#include <stdio.h>

#include <xnnpack.h>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <unordered_map>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/kernels/AvgPool.h"
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
void AvgPool(const size_t x_id, const size_t batch_size,
             const size_t input_height, const size_t input_width,
             const size_t filter_height, const size_t filter_width,
             size_t pad_top, size_t pad_right, size_t pad_bottom,
             size_t pad_left, const size_t stride_height,
             const size_t stride_width, const size_t channels,
             const size_t out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info(out_id);

  const float* x_buf = reinterpret_cast<float*>(x_info.memory_offset);
  float* out_buf = reinterpret_cast<float*>(out_info.memory_offset);

  // Implementation for a 1x1 filter (identity). xnnpack does not support 1x1
  // AvgPool
  if (filter_width == 1 && filter_height == 1) {
    // Early bailout for the identity case to use memcpy for efficiency.
    if (stride_width == 1 && stride_height == 1) {
      std::memcpy(out_buf, x_buf, out_info.size * sizeof(*x_buf));
      return;
    }

    // Values per row and column as determined by the stride size.
    // ceil(input_height / stride_height) instead of floor because strides do
    // not guarantee that more than one value is available.
    // e.g. a stride of 3 would 'partition' range(1, 10) into
    // [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
    // and would include 10 in the output: [1, 4, 7, 10]
    size_t vals_per_col = (input_height + stride_height - 1) / stride_height;
    size_t vals_per_row = (input_width + stride_width - 1) / stride_width;

    size_t x_batch_vals_count = input_width * input_height;
    size_t out_batch_vals_count = vals_per_row * vals_per_col;

    // Copy values specified by the strides.
    // Only NHWC is currently supported.
    //printf("Channels: %zd\n", channels);
    for (size_t n = 0; n < batch_size; n++) {
      for (size_t h = 0; h < vals_per_col; h++) {
        for (size_t w = 0; w < vals_per_row; w++) {
          for (size_t c = 0; c < channels; c++) {
            size_t x_n_index = n * x_batch_vals_count;
            size_t x_hw_index = h * stride_height * input_width
                                + w * stride_width;
            size_t x_nhw_index = x_n_index + x_hw_index;
            size_t x_nhwc_index = c + channels * x_nhw_index;

            size_t out_n_index = n * out_batch_vals_count;
            size_t out_hw_index = h * vals_per_row + w;
            size_t out_nhw_index = out_n_index + out_hw_index;
            size_t out_nhwc_index = c + channels * out_nhw_index;

            //printf("n: %zd, h: %zd, w: %zd, c: %zd, x_index: %zd, out_index: %zd\n", n, h, w, c, x_index, out_index);
            out_buf[out_nhwc_index] = x_buf[x_nhwc_index];
          }
        }
      }
    }
    return;
  }

  xnn_operator_t avg_pool_op = nullptr;

  const uint32_t flags = 0;
  const size_t input_pixel_stride = channels;
  const size_t output_pixel_stride = channels;

  OperatorCacheKey cache_key = {
      pad_top,       pad_right,          pad_bottom,          pad_left,
      filter_height, filter_width,       stride_height,       stride_width,
      channels,      input_pixel_stride, output_pixel_stride, flags};

  auto operator_cache_idx = operator_cache.find(cache_key);

  if (operator_cache_idx == operator_cache.end()) {
    const float output_min = -std::numeric_limits<float>::infinity();
    const float output_max = std::numeric_limits<float>::infinity();

    xnn_status status = xnn_create_average_pooling2d_nhwc_f32(
        pad_top, pad_right, pad_bottom, pad_left, filter_height, filter_width,
        stride_height, stride_width, channels, input_pixel_stride,
        output_pixel_stride, output_min, output_max, flags, &avg_pool_op);

    if (status != xnn_status_success) {
      util::warn(
          "XNN status for xnn_create_average_pooling2d_nhwc_f32 is not "
          "successful. ",
          "Got status %d. Use -c dbg to see XNN logs.", status);
      return;
    }

    operator_cache.emplace(cache_key, avg_pool_op);

    tfjs::backend::xnn_operator_count++;
  } else {
    avg_pool_op = operator_cache_idx->second;
  }

  xnn_status status = xnn_setup_average_pooling2d_nhwc_f32(
      avg_pool_op, batch_size, input_height, input_width, x_buf, out_buf,
      tfjs::backend::threadpool);
  if (status != xnn_status_success) {
    util::warn(
        "XNN status for xnn_setup_average_pooling2d_nhwc_f32 is not "
        "successful. "
        "Got status %d. Use -c dbg to see XNN logs.",
        status);
    return;
  }

  xnn_run_operator(avg_pool_op, tfjs::backend::threadpool);
}
}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
