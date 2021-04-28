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

#include "tfjs-backend-wasm/src/cc/conv2d_impl.h"

#include <xnnpack.h>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/leakyrelu_impl.h"
#include "tfjs-backend-wasm/src/cc/prelu_impl.h"
#include "tfjs-backend-wasm/src/cc/sigmoid_impl.h"
#include "tfjs-backend-wasm/src/cc/transpose_impl.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace {
// We use std::tuple as the cache key as it implements the compare operator
// needed for std::map.
typedef std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t,
                   size_t, size_t, size_t, size_t, size_t, size_t, size_t,
                   size_t, size_t, size_t, size_t, size_t, float, float>
    OperatorCacheKey;

struct CachedInfo {
  xnn_operator_t op;
  std::vector<float> transposed_filter;
};

// The operator cache maps the cache key to the xnn_operator_t instantiated for
// this set of arguments to the xnn_operator.
std::map<OperatorCacheKey, CachedInfo> operator_cache;

// Maps a filter id to a list of operator cache keys that this filter belongs
// to.
std::unordered_map<size_t, std::vector<OperatorCacheKey>>
    filter_operator_cache_key_map;

// Maps a bias id to a list of operator cache keys that this filter belongs
// to.
std::unordered_map<size_t, std::vector<OperatorCacheKey>>
    bias_operator_cache_key_map;

void erase_from_cache(const size_t tensor_id,
                      std::unordered_map<size_t, std::vector<OperatorCacheKey>>&
                          operator_cache_key_map) {
  auto operator_cache_keys_idx = operator_cache_key_map.find(tensor_id);
  if (operator_cache_keys_idx != operator_cache_key_map.end()) {
    std::vector<OperatorCacheKey>& operator_cache_keys =
        operator_cache_keys_idx->second;
    for (auto& operator_cache_key : operator_cache_keys) {
      auto operator_cache_key_idx = operator_cache.find(operator_cache_key);
      if (operator_cache_key_idx != operator_cache.end()) {
        auto& cached_info = operator_cache_key_idx->second;
        xnn_delete_operator(cached_info.op);
        tfjs::backend::xnn_operator_count--;

        operator_cache.erase(operator_cache_key);
      }
    }
    operator_cache_key_map.erase(tensor_id);
  }
}

void delete_xnn_operators(size_t tensor_id) {
  erase_from_cache(tensor_id, filter_operator_cache_key_map);
  erase_from_cache(tensor_id, bias_operator_cache_key_map);
}

void associate_tensor_with_key(
    const size_t tensor_id, const OperatorCacheKey& cache_key,
    std::unordered_map<size_t, std::vector<OperatorCacheKey>>&
        operator_cache_key_map) {
  auto cache_keys_idx = operator_cache_key_map.find(tensor_id);
  if (cache_keys_idx == operator_cache_key_map.end()) {
    std::vector<OperatorCacheKey> cache_keys = {cache_key};
    operator_cache_key_map.emplace(tensor_id, std::move(cache_keys));
    tfjs::backend::register_disposal_callback(tensor_id, *delete_xnn_operators);

  } else {
    auto& cache_keys = operator_cache_key_map.at(tensor_id);
    cache_keys.emplace_back(cache_key);
  }
}

}  // namespace

namespace tfjs {
namespace wasm {

void conv2d(const size_t x_id, const size_t batch_size,
            const size_t input_height, const size_t input_width,
            const size_t filter_id, const size_t filter_height,
            const size_t filter_width, const size_t bias_id, size_t pad_top,
            size_t pad_right, size_t pad_bottom, size_t pad_left,
            const bool is_same_pad, const size_t dilation_height,
            const size_t dilation_width, const size_t stride_height,
            const size_t stride_width, const size_t input_channels,
            const size_t output_channels, const bool is_depthwise,
            const FusableActivation activation, const size_t prelu_weights_id,
            const float leakyrelu_alpha, const size_t out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& filter_info = backend::get_tensor_info(filter_id);
  auto& out_info = backend::get_tensor_info_out(out_id);

  const float* x_buf = x_info.f32();
  const float* filter_buf = filter_info.f32();
  const float* bias_buf = nullptr;
  if (bias_id != 0) {
    bias_buf = backend::get_tensor_info_out(bias_id).f32();
  }

  float* out_buf = out_info.f32_write();
  std::vector<float> intermediate_output;

  if (prelu_weights_id != 0 || activation == FusableActivation::LEAKYRELU) {
    intermediate_output.resize(out_info.size);
    out_buf = intermediate_output.data();
  }

  xnn_operator_t conv2d_op = nullptr;

  size_t flags = 0;
  if (is_same_pad) {
    pad_top = 0, pad_right = 0, pad_bottom = 0, pad_left = 0;
    flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
  }

  size_t groups;
  size_t group_input_channels;
  size_t group_output_channels;
  const size_t input_pixel_stride = input_channels;
  const size_t output_pixel_stride = output_channels;
  if (is_depthwise) {
    groups = input_channels;
    group_input_channels = 1;
    group_output_channels = output_channels / input_channels;
    flags |= XNN_FLAG_DEPTHWISE_CONVOLUTION;
  } else {
    groups = 1;
    group_input_channels = input_channels;
    group_output_channels = output_channels;
  }

  FusableActivation clamp_method = activation;
  if (activation == FusableActivation::PRELU ||
      activation == FusableActivation::LEAKYRELU) {
    clamp_method = FusableActivation::LINEAR;
  }

  float output_min = -std::numeric_limits<float>::infinity();
  float output_max = std::numeric_limits<float>::infinity();

  if (activation == FusableActivation::RELU) {
    output_min = 0;
  } else if (activation == FusableActivation::RELU6) {
    output_min = 0;
    output_max = 6;
  }

  OperatorCacheKey cache_key = {pad_top,
                                pad_right,
                                pad_bottom,
                                pad_left,
                                filter_height,
                                filter_width,
                                stride_height,
                                stride_width,
                                dilation_height,
                                dilation_width,
                                groups,
                                group_input_channels,
                                group_output_channels,
                                input_pixel_stride,
                                output_pixel_stride,
                                clamp_method,
                                filter_id,
                                bias_id,
                                flags,
                                output_min,
                                output_max};

  auto operator_cache_idx = operator_cache.find(cache_key);
  if (operator_cache_idx == operator_cache.end()) {
    // This lives outside the if statement so the data survives the scope.
    std::vector<float> transposed_filter;

    const float* filter_xnn;
    if (is_depthwise) {
      // For depthwiseConv2d, xnn pack and TensorFlow expect the same weights
      // layout:
      //   [filter_height, filter_width, input_channels, channel_multiplier]
      filter_xnn = filter_buf;
    } else {
      // For regular conv2d, xnn pack expects weights layed out like:
      //   [output_channels, filter_height, filter_width, input_channels]
      // TensorFlow has weights layed out like:
      //   [filter_height, filter_width, input_channels, output_channels]
      // This can be transposed with a 2d transpose to move output_channels to
      // the outer most dimension.
      transposed_filter.resize(filter_info.size);
      std::vector<size_t> filter_shape = {
          filter_height * filter_width * input_channels, output_channels};
      std::vector<size_t> perm = {1, 0};

      transpose(filter_buf, filter_shape, perm, transposed_filter.data());

      filter_xnn = transposed_filter.data();
    }

    xnn_status status = xnn_create_convolution2d_nhwc_f32(
        pad_top, pad_right, pad_bottom, pad_left, filter_height, filter_width,
        stride_height, stride_width, dilation_height, dilation_width, groups,
        group_input_channels, group_output_channels, input_pixel_stride,
        output_pixel_stride, filter_xnn, bias_buf, output_min, output_max,
        flags, &conv2d_op);
    if (status != xnn_status_success) {
      util::warn(
          "XNN status for xnn_create_convolution2d_nhwc_f32 is not successful. "
          "Got status %d. Use -c dbg to see XNN logs.",
          status);
    }

    operator_cache.emplace(
        cache_key,
        // Move ownership of the transposed filter to the cache map.
        CachedInfo{conv2d_op, std::move(transposed_filter)});

    associate_tensor_with_key(filter_id, cache_key,
                              filter_operator_cache_key_map);
    if (bias_id != 0) {
      associate_tensor_with_key(bias_id, cache_key,
                                bias_operator_cache_key_map);
    }

    tfjs::backend::xnn_operator_count++;
  } else {
    conv2d_op = operator_cache_idx->second.op;
  }

  xnn_status status = xnn_setup_convolution2d_nhwc_f32(
      conv2d_op, batch_size, input_height, input_width, x_buf, out_buf,
      tfjs::backend::threadpool);
  if (status != xnn_status_success) {
    util::warn(
        "XNN status for xnn_setup_convolution2d_nhwc_f32 is not successful. "
        "Got status %d. Use -c dbg to see XNN logs.",
        status);
  }

  xnn_run_operator(conv2d_op, tfjs::backend::threadpool);

  if (activation == FusableActivation::PRELU) {
    prelu(out_buf, out_info.size, prelu_weights_id, out_id);
  }
  if (activation == FusableActivation::LEAKYRELU) {
    leakyrelu(out_buf, out_info.size, leakyrelu_alpha, out_id);
  }
  if (activation == FusableActivation::SIGMOID) {
    sigmoid(out_buf, out_info.size, out_id);
  }
}

}  // namespace wasm
}  // namespace tfjs
