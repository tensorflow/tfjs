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

#include <cstddef>

#include "src/cc/backend.h"
#include "src/cc/util.h"

const size_t kBlockSize = 48;

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void BatchMatMul(const size_t a_id, const size_t* a_shape_ptr,
                 const size_t a_shape_len, const size_t b_id,
                 const size_t* b_shape_ptr, const size_t b_shape_len,
                 const bool transpose_a, const bool transpose_b,
                 const size_t out_id) {
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

  auto& a_info = backend::get_tensor_info(a_id);
  auto& b_info = backend::get_tensor_info(b_id);
  auto& out_info = backend::get_tensor_info_out(out_id);

  const float* a_buf = a_info.f32();
  const float* b_buf = b_info.f32();
  float* out_buf = out_info.f32_write();

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
                sum +=
                    a_buf[b * a_batch + i * a_outer_step + k * a_inner_step] *
                    b_buf[k * b_inner_step + j * b_outer_step + b * b_batch];
              }
              out_buf[b * size + (i * right_dim + j)] += sum;
            }
          }
        }
      }
    }
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
