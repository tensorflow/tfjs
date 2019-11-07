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

#include "src/cc/kernels/Conv2D.h"

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
#include "src/cc/transpose_impl.h"
#include "src/cc/util.h"

namespace {
// These integer values are keys to creating the conv2d operator. We use
// std::array instead of a vanilla array as it implements the compare operator
// needed for std::map.
typedef std::array<int, 15> OperatorCacheKey;

// The operator cache maps the cache key to the xnn_operator_t instantiated for
// this set of arguments to the xnn_operator.
std::map<OperatorCacheKey, xnn_operator_t> operator_cache;

// Maps a filter id to a list of operator cache keys that this filter belongs
// to.
std::unordered_map<int, std::vector<OperatorCacheKey>>
    filter_operator_cache_key_map;

void delete_xnn_operators(int filter_id) {
  std::vector<OperatorCacheKey> operator_cache_keys =
      filter_operator_cache_key_map[filter_id];
  for (auto& operator_cache_key : operator_cache_keys) {
    auto& conv2d_op = operator_cache[operator_cache_key];
    operator_cache.erase(operator_cache_key);
    tfjs::backend::xnn_operator_count--;
  }
  filter_operator_cache_key_map.erase(filter_id);
}
}  // namespace

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void Conv2D(const int x_id, const int batch_size, const int input_height,
            const int input_width, const int filter_id, const int filter_height,
            const int filter_width, int pad_top, int pad_right, int pad_bottom,
            int pad_left, const int is_same_pad, const int dilation_height,
            const int dilation_width, const int stride_height,
            const int stride_width, const int input_channels,
            const int output_channels, const int out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& filter_info = backend::get_tensor_info(filter_id);
  auto& out_info = backend::get_tensor_info_out(out_id);

  const float* x_buf = x_info.f32();
  const float* filter_buf = filter_info.f32();
  float* out_buf = out_info.f32_write();

  xnn_operator_t conv2d_op = nullptr;

  int flags = 0;
  if (is_same_pad) {
    pad_top = 0, pad_right = 0, pad_bottom = 0, pad_left = 0;
    flags = XNN_FLAG_TENSORFLOW_SAME_PADDING;
  }

  const int groups = 1;

  OperatorCacheKey cache_key = {pad_top,         pad_right,     pad_bottom,
                                pad_left,        filter_height, filter_width,
                                stride_height,   stride_width,  dilation_height,
                                dilation_width,  groups,        input_channels,
                                output_channels, filter_id,     flags};

  auto operator_cache_idx = operator_cache.find(cache_key);
  if (operator_cache_idx == operator_cache.end()) {
    float output_min = -std::numeric_limits<float>::infinity();
    float output_max = std::numeric_limits<float>::infinity();

    // xnn pack expects weights layed out like:
    //   [output_channels, filter_height, filter_width, input_channels]
    // TensorFlow has weights layed out like:
    //   [filter_height, filter_width, input_channels, output_channels]
    // This can be transposed with a 2d transpose to move output_channels to the
    // outer most dimension.
    std::vector<float> transposed_filter(filter_info.size);

    const std::vector<int> filter_shape = {
        filter_height * filter_width * input_channels, output_channels};
    const std::vector<int> perm = {1, 0};
    tfjs::wasm::transpose(filter_buf, filter_shape, perm,
                          transposed_filter.data());

    const float* bias_buf = nullptr;
    xnn_status status = xnn_create_convolution2d_nhwc_f32(
        pad_top, pad_right, pad_bottom, pad_left, filter_height, filter_width,
        stride_height, stride_width, dilation_height, dilation_width, groups,
        input_channels /* group_input_channels */,
        output_channels /* group_output_channels */,
        input_channels /* input_pixel_stride */,
        output_channels /* output_pixel_stride */, transposed_filter.data(),
        bias_buf, output_min, output_max, flags, &conv2d_op);
    if (status != xnn_status_success) {
      util::warn(
          "XNN status for xnn_create_convolution2d_nhwc_f32 is not successful. "
          "Got status %d. Use -c dbg to see XNN logs.",
          status);
    }

    operator_cache.emplace(cache_key, conv2d_op);

    auto cache_keys_idx = filter_operator_cache_key_map.find(filter_id);
    if (cache_keys_idx == filter_operator_cache_key_map.end()) {
      std::vector<OperatorCacheKey> cache_keys = {cache_key};
      filter_operator_cache_key_map.emplace(filter_id, std::move(cache_keys));
      backend::register_disposal_callback(filter_id, *delete_xnn_operators);

    } else {
      auto& cache_keys = filter_operator_cache_key_map.at(filter_id);
      cache_keys.emplace_back(cache_key);
    }

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
