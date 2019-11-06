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

  int out_shape[3];
  out_shape[0] = x_shape[perm[0]];
  out_shape[1] = x_shape[perm[1]];
  out_shape[2] = x_shape[perm[2]];
  const int out_stride1 = out_shape[1] * out_shape[2];
  const int out_stride2 = out_shape[2];

  for (int i1 = 0; i1 < out_shape[0]; ++i1) {
    for (int i2 = 0; i2 < out_shape[1]; ++i2) {
      for (int i3 = 0; i3 < out_shape[2]; ++i3) {
        const int i = tfjs::util::offset(i1, i2, i3, 0, p1, p2, p3);
        const int o = tfjs::util::offset(i1, i2, i3, out_stride1, out_stride2);
        out_data[o] = x_data[i];
      }
    }
  }
}

// Optimized transpose 4D. For reference see `tranpose_3d`.
template <typename T>
void transpose_4d(const T* x_data, const std::vector<int>& x_shape,
                  const std::vector<int>& perm, T* out_data) {
  int s1, s2, s3, s4;
  s1 = x_shape[0];
  s2 = x_shape[1];
  s3 = x_shape[2];
  s4 = x_shape[3];

  int p1, p2, p3, p4;
  if (perm[0] == 3) {
    p1 = 1;
  } else if (perm[1] == 3) {
    p2 = 1;
  } else if (perm[2] == 3) {
    p3 = 1;
  } else {
    p4 = 1;
  }

  if (perm[0] == 2) {
    p1 = s4;
  } else if (perm[1] == 2) {
    p2 = s4;
  } else if (perm[2] == 2) {
    p3 = s4;
  } else {
    p4 = s4;
  }

  if (perm[0] == 1) {
    p1 = s3 * s4;
  } else if (perm[1] == 1) {
    p2 = s3 * s4;
  } else if (perm[2] == 1) {
    p3 = s3 * s4;
  } else {
    p4 = s3 * s4;
  }

  if (perm[0] == 0) {
    p1 = s2 * s3 * s4;
  } else if (perm[1] == 0) {
    p2 = s2 * s3 * s4;
  } else if (perm[2] == 0) {
    p3 = s2 * s3 * s4;
  } else {
    p4 = s2 * s3 * s4;
  }

  int out_shape[4];
  out_shape[0] = x_shape[perm[0]];
  out_shape[1] = x_shape[perm[1]];
  out_shape[2] = x_shape[perm[2]];
  out_shape[3] = x_shape[perm[3]];
  const int out_stride1 = out_shape[1] * out_shape[2] * out_shape[3];
  const int out_stride2 = out_shape[2] * out_shape[3];
  const int out_stride3 = out_shape[3];

  for (int i1 = 0; i1 < out_shape[0]; ++i1) {
    for (int i2 = 0; i2 < out_shape[1]; ++i2) {
      for (int i3 = 0; i3 < out_shape[2]; ++i3) {
        for (int i4 = 0; i4 < out_shape[3]; ++i4) {
          const int i = tfjs::util::offset(i1, i2, i3, i4, 0, p1, p2, p3, p4);
          const int o = tfjs::util::offset(i1, i2, i3, i4, out_stride1,
                                           out_stride2, out_stride3);
          out_data[o] = x_data[i];
        }
      }
    }
  }
}

// Generic transpose implementation for n-dim tensors.
template <typename T>
void slow_transpose_nd(const T* x_data, const std::vector<int>& x_shape,
                       const std::vector<int>& perm, T* out_data) {
  const int size = tfjs::util::size_from_shape(x_shape);
  const auto x_strides = tfjs::util::compute_strides(x_shape);
  std::vector<int> out_shape(x_shape.size());
  for (int i = 0; i < x_shape.size(); ++i) {
    out_shape[i] = x_shape[perm[i]];
  }
  const auto out_strides = tfjs::util::compute_strides(out_shape);

  for (int i = 0; i < size; ++i) {
    const auto loc = tfjs::util::offset_to_loc(i, x_strides);

    // Permute location.
    std::vector<int> new_loc(loc.size());
    for (int i = 0; i < loc.size(); ++i) {
      new_loc[i] = loc[perm[i]];
    }

    const int new_i = tfjs::util::loc_to_offset(new_loc, out_strides);
    out_data[new_i] = x_data[i];
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
  } else if (x_shape.size() == 4) {
    transpose_4d(x_data, x_shape, perm, out_data);
  } else {
    slow_transpose_nd(x_data, x_shape, perm, out_data);
  }
}

template <typename T>
void transpose(const T* x_data, const std::vector<int>& x_shape,
               const std::vector<int>& perm, T* out_data) {
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
void Transpose(const int x_id, const int* x_shape_ptr, const int x_shape_length,
               const DType dtype, const int out_id, int* perm_ptr,
               const int perm_length) {
  auto x_shape = std::vector<int>(x_shape_ptr, x_shape_ptr + x_shape_length);
  auto perm = std::vector<int>(perm_ptr, perm_ptr + perm_length);
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info_out(out_id);

  switch (dtype) {
    case DType::float32:
      transpose<float>(x_info.f32(), x_shape, perm, out_info.f32_write());
      break;
    case DType::int32:
      transpose<int>(x_info.i32(), x_shape, perm, out_info.i32_write());
      break;
    case DType::boolean:
      transpose<bool>(x_info.b(), x_shape, perm, out_info.b_write());
      break;
    default:
      util::warn("Transpose for tensor id %d failed. Unknown dtype %d", x_id,
                 dtype);
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
