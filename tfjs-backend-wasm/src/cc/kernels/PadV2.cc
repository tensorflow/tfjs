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

// Optimized transpose 2D that uses direct pointer arithmetic instead of bracket
// indexing.
template <typename T>
void transpose_2d(const T* x_data, const std::vector<int>& x_shape,
                  T* out_data) {
  const int d0 = x_shape[0];
  const int d1 = x_shape[1];
  const T* input = x_data;
  for (int i = 0; i < d0; ++i) {
    T* output = out_data + i;
    for (int j = 0; j < d1; ++j) {
      *output = *input;
      output += d0;
      ++input;
    }
  }
}

template <typename T>
void transpose_impl(const T* x_data, const std::vector<int>& x_shape,
                    const std::vector<int>& perm, T* out_data) {
  if (x_shape.size() == 2) {
    transpose_2d(x_data, x_shape, out_data);
  } else if (x_shape.size() == 3) {
    transpose_3d(x_data, x_shape, perm, out_data);
  } else if (x_shape.size() == 4) {
    transpose_4d(x_data, x_shape, perm, out_data);
  } else {
    slow_transpose_nd(x_data, x_shape, perm, out_data);
  }
}

template <typename T>
void pad(const T* x_data, const std::vector<int>& x_shape,
         const std::vector<int>& paddings, const int constant_value,
         T* out_data) {
  std::vector<int> new_x_shape;
  std::vector<int> new_perm;
  // Try to reduce the rank of the transpose by flattening any outer-most
  // dimensions.
  const int non_flatten_size = flatten(x_shape, perm, &new_x_shape, &new_perm);
  const int total_size = tfjs::util::size_from_shape(x_shape);
  for (int offset = 0; offset < total_size; offset += non_flatten_size) {
    transpose_impl(x_data + offset, new_x_shape, new_perm, out_data + offset);
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
  auto& out_info = backend::get_tensor_info(out_id);
  switch (dtype) {
    case DType::float32:
      pad<float>(reinterpret_cast<const float*>(x_info.memory_offset), x_shape,
                 paddings, constant_value,
                 reinterpret_cast<float*>(out_info.memory_offset));
      break;
    case DType::int32:
      pad<int>(reinterpret_cast<const int*>(x_info.memory_offset), x_shape,
               perm, reinterpret_cast<int*>(out_info.memory_offset));
      break;
    case DType::boolean:
      pad<bool>(reinterpret_cast<const bool*>(x_info.memory_offset), x_shape,
                perm, reinterpret_cast<bool*>(out_info.memory_offset));
      break;
    default:
      util::warn("Transpose for tensor id %d failed. Unknown dtype % d ", x_id,
                 dtype);
  }
}

}  // namespace wasm
}  // namespace wasm
}  // namespace tfjs
