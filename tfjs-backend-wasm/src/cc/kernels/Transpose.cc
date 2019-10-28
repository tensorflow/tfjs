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

// Optimized transpose 3D. Reference:
// https://github.com/tensorflow/tensorflow/blob/87388b7b6040bbf0baa67e4ef1ddc3e930ff6edd/tensorflow/lite/kernels/internal/optimized/optimized_ops.h#L7248
template <typename T>
void transpose_3d(const T* x_data, const std::vector<int>& x_shape,
                  const std::vector<int>& perm, T* out_data) {
  int s1, s2, s3;
  s1 = x_shape[0];
  s2 = x_shape[1];
  s3 = x_shape[2];

  int p1, p2, p3;
  if (perm[0] == 2) {
    p1 = 1;
  } else if (perm[1] == 2) {
    p2 = 1;
  } else {
    p3 = 1;
  }

  if (perm[0] == 1) {
    p1 = s3;
  } else if (perm[1] == 1) {
    p2 = s3;
  } else {
    p3 = s3;
  }

  if (perm[0] == 0) {
    p1 = s2 * s3;
  } else if (perm[1] == 0) {
    p2 = s2 * s3;
  } else {
    p3 = s2 * s3;
  }

  int o_s[3];
  o_s[0] = x_shape[perm[0]];
  o_s[1] = x_shape[perm[1]];
  o_s[2] = x_shape[perm[2]];

  for (int i1 = 0; i1 < o_s[0]; ++i1) {
    for (int i2 = 0; i2 < o_s[1]; ++i2) {
      for (int i3 = 0; i3 < o_s[2]; ++i3) {
        const int i = i1 * p1 + i2 * p2 + i3 * p3;
        const int o = i1 * o_s[1] * o_s[2] + i2 * o_s[2] + i3;
        out_data[o] = x_data[i];
      }
    }
  }
}

// Flatten finds the dimensions that can be flatten, shrinks the given shapes
// and the given perm parameter to reflect the non-flatten dimensions, and
// returns the total size of the non-flatten dimensions.
//
// E.g, Given shape [2, 3, 4, 5] and perm [0,1,3,2] case,
// this method flattens the first two dimensions and returns a new shape [5,4],
// new perm [1,0] and 4*5=20 as the total size of the non-flat dims. Reference:
// https://github.com/tensorflow/tensorflow/blob/1f404fcaad58bf61a107d4fa7c4f6004168a50fa/tensorflow/lite/kernels/internal/transpose_utils.h#L42
int flatten(const std::vector<int>& x_shape, const std::vector<int>& perm,
            std::vector<int>* new_x_shape_ptr, std::vector<int>* new_perm_ptr) {
  auto& new_input_shape = *new_x_shape_ptr;
  auto& new_perm = *new_perm_ptr;

  // Calculate the total size of non-flatten dimensions.
  int num_dims_to_skip = 0;
  int rank = perm.size();
  int flat_size = tfjs::util::size_from_shape(x_shape);
  for (int i = 0; i < rank; ++i) {
    if (perm[i] == i) {
      flat_size /= x_shape[i];
      ++num_dims_to_skip;
    } else {
      break;
    }
  }
  // Shrink the shapes and re-calculate the perm parameter.
  const int new_rank = rank - num_dims_to_skip;
  new_perm.resize(new_rank);
  new_input_shape.resize(new_rank);

  for (int i = num_dims_to_skip; i < rank; ++i) {
    new_input_shape[i - num_dims_to_skip] = x_shape[i];
    new_perm[i - num_dims_to_skip] = perm[i];
  }
  for (int i = 0; i < new_rank; ++i) {
    int min_val_idx = -1;
    for (int j = 0; j < new_rank; ++j) {
      if (new_perm[j] >= i &&
          (min_val_idx == -1 || new_perm[min_val_idx] > new_perm[j])) {
        min_val_idx = j;
      }
    }
    new_perm[min_val_idx] = i;
  }
  return flat_size;
}

template <typename T>
void transpose_impl(const T* x_data, const std::vector<int>& x_shape,
                    const std::vector<int>& perm, T* out_data) {
  if (x_shape.size() == 2) {
    transpose_2d(x_data, x_shape, out_data);
  } else if (x_shape.size() == 3) {
    transpose_3d(x_data, x_shape, perm, out_data);
  } else {
    // TODO(smilkov): Add a 4D transpose.
    tfjs::util::warn("WASM Transpose kernel does not yet support rank %d",
                     x_shape.size());
  }
}

template <typename T>
void transpose(const T* x_data, const std::vector<int>& x_shape,
               const std::vector<int>& perm, T* out_data) {
  std::vector<int> new_x_shape;
  std::vector<int> new_perm;
  // Try to reduce the rank of the transposition by flattening any outer-most
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
void Transpose(int x_id, int* x_shape_ptr, int x_shape_length, int out_id,
               int* perm_ptr, int perm_length) {
  auto x_shape = std::vector<int>(x_shape_ptr, x_shape_ptr + x_shape_length);
  auto perm = std::vector<int>(perm_ptr, perm_ptr + perm_length);
  const TensorInfo x_info = backend::get_tensor_info(x_id);
  const TensorInfo out_info = backend::get_tensor_info(out_id);

  switch (x_info.dtype) {
    case DType::float32:
      transpose<float>(x_info.buf.f32, x_shape, perm, out_info.buf.f32);
      break;
    case DType::int32:
      transpose<int>(x_info.buf.i32, x_shape, perm, out_info.buf.i32);
      break;
    case DType::boolean:
      transpose<bool>(x_info.buf.b, x_shape, perm, out_info.buf.b);
      break;
    default:
      util::warn("Transpose for tensor id %d failed. Unknown dtype %d", x_id,
                 x_info.dtype);
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
