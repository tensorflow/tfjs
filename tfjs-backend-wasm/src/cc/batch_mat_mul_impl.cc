/* Copyright 2020 Google LLC. All Rights Reserved.
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

#include <xnnpack.h>
#include <algorithm>
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
#include "tfjs-backend-wasm/src/cc/elu_impl.h"
#include "tfjs-backend-wasm/src/cc/leakyrelu_impl.h"
#include "tfjs-backend-wasm/src/cc/prelu_impl.h"
#include "tfjs-backend-wasm/src/cc/sigmoid_impl.h"
#include "tfjs-backend-wasm/src/cc/util.h"

#include "tfjs-backend-wasm/src/cc/batch_mat_mul_impl.h"

const size_t kBlockSize = 48;

namespace {
// We use std::tuple as the cache key as it implements the compare operator
// needed for std::map.
typedef std::tuple<size_t, size_t, size_t> OperatorCacheKey;

// The operator cache maps the weights id to the xnn_operator_t instantiated for
// this set of weights.
std::map<OperatorCacheKey, xnn_operator_t> operator_cache;

std::unordered_map<size_t, std::vector<OperatorCacheKey>>
    b_operator_cache_key_map;

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
        auto& cached_op = operator_cache_key_idx->second;
        xnn_delete_operator(cached_op);
        tfjs::backend::xnn_operator_count--;

        operator_cache.erase(operator_cache_key);
      }
    }
    operator_cache_key_map.erase(tensor_id);
  }
}

void delete_xnn_operators(const size_t tensor_id) {
  erase_from_cache(tensor_id, b_operator_cache_key_map);
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

void xnn_matmul(const size_t a_id, const size_t* a_shape_ptr,
                const size_t a_shape_len, const size_t b_id,
                const size_t* b_shape_ptr, const size_t b_shape_len,
                const size_t out_id, const size_t bias_id,
                const float output_min, const float output_max,
                const size_t clamp_method) {
  auto& a_info = tfjs::backend::get_tensor_info(a_id);
  auto& b_info = tfjs::backend::get_tensor_info(b_id);
  auto& out_info = tfjs::backend::get_tensor_info_out(out_id);

  const float* a_buf = a_info.f32();
  const float* b_buf = b_info.f32();
  float* out_buf = out_info.f32_write();

  const float* bias_buf = nullptr;
  if (bias_id != 0) {
    bias_buf = tfjs::backend::get_tensor_info_out(bias_id).f32();
  }

  xnn_operator_t fully_connected_op = nullptr;

  OperatorCacheKey cache_key = {b_id, bias_id, clamp_method};

  // We assume b is the weights and cache the xnn operator on it.
  auto operator_cache_idx = operator_cache.find(cache_key);
  if (operator_cache_idx == operator_cache.end()) {
    const size_t input_channels = b_shape_ptr[1];
    const size_t output_channels = b_shape_ptr[2];
    const size_t input_stride = input_channels;
    const size_t output_stride = output_channels;

    // XNNPack expects b to already be transposed. TensorFlow.js doesn't do this
    // automatically so we have to tell XNNPack to do the transposing.
    const uint32_t flags = XNN_FLAG_TRANSPOSE_WEIGHTS;
    xnn_status status = xnn_create_fully_connected_nc_f32(
        input_channels, output_channels, input_stride, output_stride, b_buf,
        bias_buf, output_min, output_max, flags, &fully_connected_op);
    if (status != xnn_status_success) {
      tfjs::util::warn(
          "XNN status for xnn_create_fully_connected_nc_f32 is not successful. "
          "Got status %d. Use -c dbg to see XNN logs.",
          status);
      return;
    }

    operator_cache.insert({cache_key, fully_connected_op});

    associate_tensor_with_key(b_id, cache_key, b_operator_cache_key_map);
    if (bias_id != 0) {
      associate_tensor_with_key(bias_id, cache_key,
                                bias_operator_cache_key_map);
    }

    tfjs::backend::xnn_operator_count++;
  } else {
    fully_connected_op = operator_cache_idx->second;
  }

  const size_t batch_size = a_shape_ptr[1];
  xnn_status status =
      xnn_setup_fully_connected_nc_f32(fully_connected_op, batch_size, a_buf,
                                       out_buf, tfjs::backend::threadpool);
  if (status != xnn_status_success) {
    tfjs::util::warn(
        "XNN status for xnn_setup_fully_connected_nc_f32 is not successful. "
        "Got status %d. Use -c dbg to see XNN logs.",
        status);
    return;
  }

  xnn_run_operator(fully_connected_op, tfjs::backend::threadpool);
}

void slow_batch_matmul(const size_t a_id, const size_t* a_shape_ptr,
                       const size_t a_shape_len, const size_t b_id,
                       const size_t* b_shape_ptr, const size_t b_shape_len,
                       const bool transpose_a, const bool transpose_b,
                       const size_t out_id, const size_t bias_id,
                       const float output_min, const float output_max) {
  const size_t shared_dim = transpose_a ? a_shape_ptr[1] : a_shape_ptr[2];
  const size_t left_dim = transpose_a ? a_shape_ptr[2] : a_shape_ptr[1];
  const size_t right_dim = transpose_b ? b_shape_ptr[1] : b_shape_ptr[2];
  const size_t batch_dim = a_shape_ptr[0];

  std::vector<size_t> a_shape(a_shape_ptr, a_shape_ptr + a_shape_len);
  std::vector<size_t> b_shape(b_shape_ptr, b_shape_ptr + b_shape_len);
  const std::vector<size_t> a_strides = tfjs::util::compute_strides(a_shape);
  const std::vector<size_t> b_strides = tfjs::util::compute_strides(b_shape);

  size_t a_batch = a_strides[0];
  size_t a_outer_step, a_inner_step;
  if (transpose_a) {
    a_outer_step = 1;
    a_inner_step = a_strides[1];
  } else {
    a_outer_step = a_strides[1];
    a_inner_step = 1;
  }
  size_t b_batch = b_strides[0];
  size_t b_outer_step, b_inner_step;
  if (transpose_b) {
    b_outer_step = b_strides[1];
    b_inner_step = 1;
  } else {
    b_outer_step = 1;
    b_inner_step = b_strides[1];
  }

  auto& a_info = tfjs::backend::get_tensor_info(a_id);
  auto& b_info = tfjs::backend::get_tensor_info(b_id);
  auto& out_info = tfjs::backend::get_tensor_info_out(out_id);

  const float* a_buf = a_info.f32();
  const float* b_buf = b_info.f32();
  float* out_buf = out_info.f32_write();

  const float* bias_buf = nullptr;
  size_t bias_buf_size = 0;
  if (bias_id != 0) {
    auto& bias_info = tfjs::backend::get_tensor_info_out(bias_id);
    bias_buf = bias_info.f32();
    bias_buf_size = bias_info.size;
  }

  const size_t size = left_dim * right_dim;

  // Zero out the output buffer because it might have been used before.
  std::fill(out_buf, out_buf + batch_dim * size, 0);

  for (size_t b = 0; b < batch_dim; ++b) {
    for (size_t i0 = 0; i0 < left_dim; i0 += kBlockSize) {
      for (size_t j0 = 0; j0 < right_dim; j0 += kBlockSize) {
        for (size_t k0 = 0; k0 < shared_dim; k0 += kBlockSize) {
          // for when kBlockSize doesn't evenly divide the input
          const size_t i_block = std::min(i0 + kBlockSize, left_dim);
          const size_t j_block = std::min(j0 + kBlockSize, right_dim);
          const size_t k_block = std::min(k0 + kBlockSize, shared_dim);

          for (size_t i = i0; i < i_block; ++i) {
            for (size_t j = j0; j < j_block; ++j) {
              float sum = 0.0;

              for (size_t k = k0; k < k_block; ++k) {
                const size_t batch_offset_a =
                    std::min(b, a_shape[0] - 1) * a_batch;
                const size_t batch_offset_b =
                    std::min(b, b_shape[0] - 1) * b_batch;
                sum +=
                    a_buf[batch_offset_a + i * a_outer_step +
                          k * a_inner_step] *
                    b_buf[k * b_inner_step + j * b_outer_step + batch_offset_b];
              }
              size_t innermost_dim = i * right_dim + j;
              size_t out_buf_index = b * size + innermost_dim;
              float current = out_buf[out_buf_index];

              float bias_val = 0;
              if (bias_id != 0) {
                // Handles 1D broadcasting.
                size_t bias_index = std::min(innermost_dim, bias_buf_size - 1);

                bias_val = bias_buf[bias_index];
              }

              out_buf[out_buf_index] = std::max(
                  std::min(current + sum + bias_val, output_max), output_min);
            }
          }
        }
      }
    }
  }
}
}  // namespace

namespace tfjs {
namespace wasm {
void fused_batch_mat_mul(const size_t a_id, const size_t* a_shape_ptr,
                         const size_t a_shape_len, const size_t b_id,
                         const size_t* b_shape_ptr, const size_t b_shape_len,
                         const bool transpose_a, const bool transpose_b,
                         const FusableActivation activation,
                         const size_t bias_id, const size_t prelu_weights_id,
                         const float leakyrelu_alpha, const size_t out_id) {
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

  if (!transpose_a && !transpose_b && a_shape_ptr[0] == 1 &&
      b_shape_ptr[0] == 1) {
    xnn_matmul(a_id, a_shape_ptr, a_shape_len, b_id, b_shape_ptr, b_shape_len,
               out_id, bias_id, output_min, output_max, clamp_method);
  } else {
    slow_batch_matmul(a_id, a_shape_ptr, a_shape_len, b_id, b_shape_ptr,
                      b_shape_len, transpose_a, transpose_b, out_id, bias_id,
                      output_min, output_max);
  }

  auto& out_info = backend::get_tensor_info_out(out_id);
  float* out_buf = out_info.f32_write();
  if (activation == FusableActivation::PRELU) {
    prelu(out_buf, out_info.size, prelu_weights_id, out_id);
  } else if (activation == FusableActivation::LEAKYRELU) {
    leakyrelu(out_buf, out_info.size, leakyrelu_alpha, out_id);
  } else if (activation == FusableActivation::SIGMOID) {
    sigmoid(out_buf, out_info.size, out_id);
  } else if (activation == FusableActivation::ELU) {
    elu(out_buf, out_info.size, out_id);
  }
}
}  // namespace wasm
}  // namespace tfjs
