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
#include <cstddef>
#include <cstring>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/kernels/PadV2.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace {
// We use std::tuple as the cache key as it implements the compare operator
// needed for std::map.
typedef std::tuple<float, uint32_t> OperatorCacheKey;

// The operator cache maps the weights id to the xnn_operator_t instantiated for
// this set of weights.
std::map<OperatorCacheKey, xnn_operator_t> operator_cache;

}  // namespace

namespace {

using tfjs::util::compute_strides;
using tfjs::util::loc_to_offset;
using tfjs::util::offset_to_loc;
using tfjs::util::size_from_shape;

// Reference:
// https://github.com/tensorflow/tensorflow/blob/a4190274142b70f3d5b7a27f93fc14abc7448240/tensorflow/lite/kernels/internal/optimized/optimized_ops.h#L4873
template <typename T>
void typed_memset(void* ptr, T value, size_t num) {
  // Optimization for common cases where memset() will suffice.
  if (value == 0 || std::is_same<T, uint8_t>::value) {
    memset(ptr, value, num * sizeof(T));
  } else {
    // Default implementation for cases where memset() will not preserve the
    // bytes, e.g., typically when sizeof(T) > sizeof(uint8_t).
    char* pos = static_cast<char*>(ptr);
    for (size_t i = 0; i < num; ++i) {
      memcpy(pos, &value, sizeof(T));
      pos += sizeof(T);
    }
  }
}

template <typename T>
void pad_4d(const T* x_data, size_t x_shape[4], size_t paddings[8],
            const T pad_value, size_t out_shape[4], T* out_data) {
  const size_t left_b_pad = paddings[0];
  const size_t right_b_pad = paddings[1];
  const size_t left_h_pad = paddings[2];
  const size_t right_h_pad = paddings[3];
  const size_t left_w_pad = paddings[4];
  const size_t right_w_pad = paddings[5];
  const size_t left_d_pad = paddings[6];
  const size_t right_d_pad = paddings[7];

  const size_t batch = x_shape[0];
  const size_t height = x_shape[1];
  const size_t width = x_shape[2];
  const size_t depth = x_shape[3];

  const size_t out_height = out_shape[1];
  const size_t out_width = out_shape[2];
  const size_t out_depth = out_shape[3];

  T* out_offset = out_data;
  const T* x_offset = x_data;

  if (left_b_pad != 0) {
    typed_memset<T>(out_offset, pad_value,
                    left_b_pad * out_height * out_width * out_depth);
    out_offset += left_b_pad * out_height * out_width * out_depth;
  }
  for (size_t b = 0; b < batch; ++b) {
    if (left_h_pad != 0) {
      typed_memset<T>(out_offset, pad_value,
                      left_h_pad * out_width * out_depth);
      out_offset += left_h_pad * out_width * out_depth;
    }
    for (size_t h = 0; h < height; ++h) {
      if (left_w_pad != 0) {
        typed_memset<T>(out_offset, pad_value, left_w_pad * out_depth);
        out_offset += left_w_pad * out_depth;
      }
      for (size_t w = 0; w < width; ++w) {
        if (left_d_pad != 0) {
          typed_memset<T>(out_offset, pad_value, left_d_pad);
          out_offset += left_d_pad;
        }
        memcpy(out_offset, x_offset, depth * sizeof(T));
        x_offset += depth;
        out_offset += depth;

        if (right_d_pad != 0) {
          typed_memset<T>(out_offset, pad_value, right_d_pad);
          out_offset += right_d_pad;
        }
      }
      if (right_w_pad != 0) {
        typed_memset<T>(out_offset, pad_value, right_w_pad * out_depth);
        out_offset += right_w_pad * out_depth;
      }
    }
    if (right_h_pad != 0) {
      typed_memset<T>(out_offset, pad_value,
                      right_h_pad * out_width * out_depth);
      out_offset += right_h_pad * out_width * out_depth;
    }
  }
  if (right_b_pad != 0) {
    typed_memset<T>(out_offset, pad_value,
                    right_b_pad * out_height * out_width * out_depth);
  }
}

// Generic pad implementation for n-dim tensors.
template <typename T>
void slow_pad_nd(const T* x_data, const std::vector<size_t>& x_shape,
                 const std::vector<size_t>& pre_paddings,
                 const std::vector<size_t>& post_paddings, const T pad_value,
                 T* out_data) {
  const size_t rank = x_shape.size();
  std::vector<size_t> out_shape(rank);
  for (size_t i = 0; i < rank; ++i) {
    const size_t pad_left = pre_paddings[i];
    const size_t pad_right = post_paddings[i];
    out_shape[i] = x_shape[i] + pad_left + pad_right;
  }
  const auto& in_strides = compute_strides(x_shape);
  const auto& out_strides = compute_strides(out_shape);
  const size_t in_size = size_from_shape(x_shape);
  const size_t out_size = size_from_shape(out_shape);

  typed_memset<T>(out_data, pad_value, out_size);

  for (size_t i = 0; i < in_size; ++i) {
    auto out_loc = offset_to_loc(i, in_strides);
    for (size_t j = 0; j < rank; ++j) {
      out_loc[j] += pre_paddings[j];
    }
    const size_t out_offset = loc_to_offset(out_loc, out_strides);
    out_data[out_offset] = x_data[i];
  }
}

template <typename T>
void pad(const T* x_data, const std::vector<size_t>& x_shape,
         const std::vector<size_t>& pre_paddings,
         const std::vector<size_t>& post_paddings, const T pad_value,
         T* out_data) {
  const size_t rank = x_shape.size();
  if (rank <= 4) {
    // Expand the shape to be 4d.
    size_t x_shape_4d[4];
    size_t paddings_4d[8];
    size_t out_shape_4d[4];
    const size_t rank_shift = 4 - rank;
    for (size_t i = 0; i < rank_shift; ++i) {
      x_shape_4d[i] = 1;
      out_shape_4d[i] = 1;
      paddings_4d[i * 2] = 0;
      paddings_4d[i * 2 + 1] = 0;
    }

    for (size_t i = 0; i < rank; ++i) {
      size_t j = i + rank_shift;
      const size_t pad_left = pre_paddings[i];
      const size_t pad_right = post_paddings[i];
      x_shape_4d[j] = x_shape[i];
      out_shape_4d[j] = x_shape[i] + pad_left + pad_right;
      paddings_4d[j * 2] = pad_left;
      paddings_4d[j * 2 + 1] = pad_right;
    }
    pad_4d(x_data, x_shape_4d, paddings_4d, pad_value, out_shape_4d, out_data);
  } else {
    slow_pad_nd(x_data, x_shape, pre_paddings, post_paddings, pad_value,
                out_data);
  }
}

}  // namespace

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void PadV2(const size_t x_id, const size_t* x_shape_ptr,
           const size_t x_shape_length, const DType dtype,
           const size_t* pre_paddings_ptr, const size_t* post_paddings_ptr,
           const float pad_value, const size_t out_id) {
  auto x_shape = std::vector<size_t>(x_shape_ptr, x_shape_ptr + x_shape_length);
  auto pre_paddings =
      std::vector<size_t>(pre_paddings_ptr, pre_paddings_ptr + x_shape_length);
  auto post_paddings = std::vector<size_t>(post_paddings_ptr,
                                           post_paddings_ptr + x_shape_length);

  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  switch (dtype) {
    case DType::float32: {
      xnn_operator_t pad_op = nullptr;
      const uint32_t flags = 0;

      OperatorCacheKey cache_key = {pad_value, flags};

      auto operator_cache_idx = operator_cache.find(cache_key);
      if (operator_cache_idx == operator_cache.end()) {
        xnn_status status =
            xnn_create_constant_pad_nd_x32(&pad_value, flags, &pad_op);
        if (status != xnn_status_success) {
          tfjs::util::warn(
              "XNN status for xnn_create_constant_pad_nd_x32 is not "
              "successful. Got status %d. Use -c dbg to see XNN logs.",
              status);
          return;
        }

        operator_cache.insert({cache_key, pad_op});

        tfjs::backend::xnn_operator_count++;
      } else {
        pad_op = operator_cache_idx->second;
      }

      xnn_status status = xnn_setup_constant_pad_nd_x32(
          pad_op, x_shape_length, x_shape_ptr, pre_paddings_ptr,
          post_paddings_ptr, x_info.f32(), out_info.f32_write(),
          tfjs::backend::threadpool);
      if (status != xnn_status_success) {
        tfjs::util::warn(
            "XNN status for xnn_setup_constant_pad_nd_x32 is not "
            "successful. Got status %d. Use -c dbg to see XNN logs.",
            status);
        return;
      }

      xnn_run_operator(pad_op, tfjs::backend::threadpool);
      break;
    }
    case DType::int32:
      pad<int32_t>(x_info.i32(), x_shape, pre_paddings, post_paddings,
                   static_cast<int32_t>(pad_value), out_info.i32_write());
      break;
    case DType::boolean:
      pad<bool>(x_info.b(), x_shape, pre_paddings, post_paddings,
                static_cast<bool>(pad_value), out_info.b_write());
      break;
    default:
      util::warn("Pad for tensor id %d failed. Unknown dtype % d ", x_id,
                 dtype);
  }
}

}  // namespace wasm
}  // namespace wasm
}  // namespace tfjs
