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

#include "src/cc/backend.h"
#include "src/cc/util.h"

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
void pad_4d(const T* x_data, int x_shape[4], int paddings[8], const T pad_value,
            int out_shape[4], T* out_data) {
  const int left_b_pad = paddings[0];
  const int right_b_pad = paddings[1];
  const int left_h_pad = paddings[2];
  const int right_h_pad = paddings[3];
  const int left_w_pad = paddings[4];
  const int right_w_pad = paddings[5];
  const int left_d_pad = paddings[6];
  const int right_d_pad = paddings[7];

  const int batch = x_shape[0];
  const int height = x_shape[1];
  const int width = x_shape[2];
  const int depth = x_shape[3];

  const int out_height = out_shape[1];
  const int out_width = out_shape[2];
  const int out_depth = out_shape[3];

  T* out_offset = out_data;
  const T* x_offset = x_data;

  if (left_b_pad != 0) {
    typed_memset<T>(out_offset, pad_value,
                    left_b_pad * out_height * out_width * out_depth);
    out_offset += left_b_pad * out_height * out_width * out_depth;
  }
  for (int b = 0; b < batch; ++b) {
    if (left_h_pad != 0) {
      typed_memset<T>(out_offset, pad_value,
                      left_h_pad * out_width * out_depth);
      out_offset += left_h_pad * out_width * out_depth;
    }
    for (int h = 0; h < height; ++h) {
      if (left_w_pad != 0) {
        typed_memset<T>(out_offset, pad_value, left_w_pad * out_depth);
        out_offset += left_w_pad * out_depth;
      }
      for (int w = 0; w < width; ++w) {
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
void slow_pad_nd(const T* x_data, const std::vector<int>& x_shape,
                 const std::vector<int>& paddings, const T pad_value,
                 T* out_data) {
  const int rank = x_shape.size();
  std::vector<int> out_shape(rank);
  for (size_t i = 0; i < rank; ++i) {
    const int pad_left = paddings[i * 2];
    const int pad_right = paddings[i * 2 + 1];
    out_shape[i] = x_shape[i] + pad_left + pad_right;
  }
  const auto& in_strides = compute_strides(x_shape);
  const auto& out_strides = compute_strides(out_shape);
  const int in_size = size_from_shape(x_shape);
  const int out_size = size_from_shape(out_shape);

  typed_memset<T>(out_data, pad_value, out_size);

  for (size_t i = 0; i < in_size; ++i) {
    auto out_loc = offset_to_loc(i, in_strides);
    for (size_t j = 0; j < rank; ++j) {
      out_loc[j] += paddings[j * 2];
    }
    const int out_offset = loc_to_offset(out_loc, out_strides);
    out_data[out_offset] = x_data[i];
  }
}

template <typename T>
void pad(const T* x_data, const std::vector<int>& x_shape,
         const std::vector<int>& paddings, const T pad_value, T* out_data) {
  const size_t rank = x_shape.size();
  if (rank <= 4) {
    // Expand the shape to be 4d.
    int x_shape_4d[4];
    int paddings_4d[8];
    int out_shape_4d[4];
    const size_t rank_shift = 4 - rank;
    for (size_t i = 0; i < rank_shift; ++i) {
      x_shape_4d[i] = 1;
      out_shape_4d[i] = 1;
      paddings_4d[i * 2] = 0;
      paddings_4d[i * 2 + 1] = 0;
    }

    for (size_t i = 0; i < rank; ++i) {
      size_t j = i + rank_shift;
      const int pad_left = paddings[i * 2];
      const int pad_right = paddings[i * 2 + 1];
      x_shape_4d[j] = x_shape[i];
      out_shape_4d[j] = x_shape[i] + pad_left + pad_right;
      paddings_4d[j * 2] = pad_left;
      paddings_4d[j * 2 + 1] = pad_right;
    }
    pad_4d(x_data, x_shape_4d, paddings_4d, pad_value, out_shape_4d, out_data);
  } else {
    slow_pad_nd(x_data, x_shape, paddings, pad_value, out_data);
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
void PadV2(const int x_id, const int* x_shape_ptr, const int x_shape_length,
           const DType dtype, const int* paddings_ptr, const float pad_value,
           const int out_id) {
  auto x_shape = std::vector<int>(x_shape_ptr, x_shape_ptr + x_shape_length);
  const int paddings_length = x_shape_length * 2;
  auto paddings =
      std::vector<int>(paddings_ptr, paddings_ptr + paddings_length);
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  switch (dtype) {
    case DType::float32:
      pad<float>(x_info.f32(), x_shape, paddings, pad_value,
                 out_info.f32_write());
      break;
    case DType::int32:
      pad<int>(x_info.i32(), x_shape, paddings, static_cast<int>(pad_value),
               out_info.i32_write());
      break;
    case DType::boolean:
      pad<bool>(x_info.b(), x_shape, paddings, static_cast<bool>(pad_value),
                out_info.b_write());
      break;
    default:
      util::warn("Pad for tensor id %d failed. Unknown dtype % d ", x_id,
                 dtype);
  }
}

}  // namespace wasm
}  // namespace wasm
}  // namespace tfjs
