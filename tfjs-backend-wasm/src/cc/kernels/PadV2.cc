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
void pad_2d(const T* x_data, const std::vector<int>& x_shape,
            const std::vector<int>& paddings, const int pad_value,
            T* out_data) {
  const int left_b_pad = paddings[0];
  const int right_b_pad = paddings[1];
  const int left_d_pad = paddings[2];
  const int right_d_pad = paddings[3];

  const int batch = x_shape[0];
  const int depth = x_shape[1];

  T* out_offset = out_data;
  const T* x_offset = x_data;

  if (left_b_pad != 0) {
    typed_memset<T>(out_offset, pad_value, left_b_pad * depth);
    out_offset += left_b_pad * depth;
  }
  for (int b = 0; b < batch; ++b) {
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
  if (right_b_pad != 0) {
    typed_memset<T>(out_offset, pad_value, right_b_pad * depth);
  }
}

template <typename T>
void pad(const T* x_data, const std::vector<int>& x_shape,
         const std::vector<int>& paddings, const int constant_value,
         T* out_data) {
  if (x_shape.size() == 2) {
    pad_2d(x_data, x_shape, paddings, constant_value, out_data);
  } else {
    tfjs::util::warn("Padding for rank %d is not yet supported",
                     x_shape.size());
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
           const DType dtype, const int* paddings_ptr, const int constant_value,
           const int out_id) {
  auto x_shape = std::vector<int>(x_shape_ptr, x_shape_ptr + x_shape_length);
  const int paddings_length = x_shape_length * 2;
  auto paddings =
      std::vector<int>(paddings_ptr, paddings_ptr + paddings_length);
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  switch (dtype) {
    case DType::float32:
      pad<float>(x_info.f32(), x_shape, paddings, constant_value,
                 out_info.f32_write());
      break;
    case DType::int32:
      pad<int>(x_info.i32(), x_shape, paddings, constant_value,
               out_info.i32_write());
      break;
    case DType::boolean:
      pad<bool>(x_info.b(), x_shape, paddings, constant_value,
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
