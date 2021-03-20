/* Copyright 2021 Google LLC. All Rights Reserved.
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

#include <cstddef>
#include <cstring>
#include <vector>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/util.h"

// Must match enum in MirrorPad.ts
enum MirrorPaddingMode {
  REFLECT = 0,
  SYMMETRIC = 1,
};

namespace {

using tfjs::util::compute_strides;
using tfjs::util::size_from_shape;

template <typename T>
void mirror_pad_recursion(const T* x_data, const std::vector<size_t>& x_shape,
                          const std::vector<size_t>& pre_paddings,
                          const std::vector<size_t>& post_paddings,
                          const int32_t offset,
                          const std::vector<size_t>& in_strides,
                          const std::vector<size_t>& out_strides, T* out_data,
                          const size_t dim, size_t in_offset,
                          size_t out_offset) {
  const size_t depth = x_shape[dim];
  const size_t rank = x_shape.size();
  const size_t in_stride = dim == rank - 1 ? 1 : in_strides[dim];
  const size_t out_stride = dim == rank - 1 ? 1 : out_strides[dim];

  out_offset += pre_paddings[dim] * out_stride;

  if (dim == rank - 1) {
    memcpy(out_data + out_offset, x_data + in_offset, sizeof(T) * depth);
  } else {
    for (size_t i = 0; i < depth; ++i) {
      mirror_pad_recursion(x_data, x_shape, pre_paddings, post_paddings, offset,
                           in_strides, out_strides, out_data, dim + 1,
                           in_offset + in_stride * i,
                           out_offset + out_stride * i);
    }
  }

  const T* src = out_data + out_offset + offset * out_stride;
  T* dist = out_data + out_offset - out_stride;
  for (size_t i = 0; i < pre_paddings[dim]; ++i) {
    memcpy(dist, src, out_stride * sizeof(T));
    src += out_stride;
    dist -= out_stride;
  }

  out_offset += (depth - 1) * out_stride;

  src = out_data + out_offset - offset * out_stride;
  dist = out_data + out_offset + out_stride;
  for (size_t i = 0; i < post_paddings[dim]; ++i) {
    memcpy(dist, src, out_stride * sizeof(T));
    src -= out_stride;
    dist += out_stride;
  }
}

template <typename T>
void mirror_pad(const T* x_data, const std::vector<size_t>& x_shape,
                const std::vector<size_t>& pre_paddings,
                const std::vector<size_t>& post_paddings,
                const MirrorPaddingMode mode, T* out_data) {
  const size_t rank = x_shape.size();
  std::vector<size_t> out_shape(rank);
  for (size_t i = 0; i < rank; ++i) {
    const size_t pad_left = pre_paddings[i];
    const size_t pad_right = post_paddings[i];
    out_shape[i] = x_shape[i] + pad_left + pad_right;
  }
  const int32_t offset = mode == MirrorPaddingMode::REFLECT ? 1 : 0;
  std::vector<size_t> in_strides = compute_strides(x_shape);
  std::vector<size_t> out_strides = compute_strides(out_shape);

  mirror_pad_recursion(x_data, x_shape, pre_paddings, post_paddings, offset,
                       in_strides, out_strides, out_data, 0, 0, 0);
}

}  // namespace

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void MirrorPad(const size_t x_id, const size_t* x_shape_ptr,
               const size_t x_shape_length, const DType dtype,
               const size_t* pre_paddings_ptr, const size_t* post_paddings_ptr,
               const MirrorPaddingMode mode, const size_t out_id) {
  auto x_shape = std::vector<size_t>(x_shape_ptr, x_shape_ptr + x_shape_length);
  auto pre_paddings =
      std::vector<size_t>(pre_paddings_ptr, pre_paddings_ptr + x_shape_length);
  auto post_paddings = std::vector<size_t>(post_paddings_ptr,
                                           post_paddings_ptr + x_shape_length);

  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  switch (dtype) {
    case DType::float32:
      mirror_pad<float>(x_info.f32(), x_shape, pre_paddings, post_paddings,
                        mode, out_info.f32_write());
      break;
    case DType::int32:
      mirror_pad<int32_t>(x_info.i32(), x_shape, pre_paddings, post_paddings,
                          mode, out_info.i32_write());
      break;
    case DType::boolean:
      mirror_pad<bool>(x_info.b(), x_shape, pre_paddings, post_paddings, mode,
                       out_info.b_write());
      break;
    default:
      util::warn("MirrorPad for tensor id %d failed. Unknown dtype % d ", x_id,
                 dtype);
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
