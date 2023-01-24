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

#include <vector>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/scatter_impl.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace {
template <typename T>
void sparse_to_dense(const int* sparse_indices_ptr, const T* sparse_values_ptr,
                     const size_t sparse_values_rank,
                     const T* default_value_ptr, const size_t slice_rank,
                     const size_t num_updates, const size_t slice_size,
                     const std::vector<size_t>& strides_ptr,
                     const size_t output_size, T* out_buf_ptr) {
  T default_value = *default_value_ptr;
  bool sum_dupe_indices = false;
  bool update_as_scalar = sparse_values_rank == 0;
  tfjs::wasm::scatter(sparse_indices_ptr, sparse_values_ptr, slice_rank,
                      num_updates, slice_size, strides_ptr, output_size,
                      default_value, sum_dupe_indices, update_as_scalar,
                      out_buf_ptr);
}

}  // namespace

namespace tfjs {
namespace wasm {
extern "C" {
#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void SparseToDense(const size_t sparse_indices_id,
                   const size_t sparse_values_id,
                   const size_t sparse_values_rank,
                   const size_t default_value_id, const DType dtype,
                   const size_t slice_rank, const size_t num_updates,
                   const size_t slice_size, const size_t* strides_ptr,
                   const size_t output_size, const size_t out_id) {
  auto& sparse_indices_info = backend::get_tensor_info(sparse_indices_id);
  auto& sparse_values_info = backend::get_tensor_info(sparse_values_id);
  auto& default_value_info = backend::get_tensor_info(default_value_id);
  const std::vector<size_t> strides(strides_ptr, strides_ptr + slice_rank);
  const int* sparse_indices_buf = sparse_indices_info.i32();
  auto& out_info = backend::get_tensor_info_out(out_id);

  switch (dtype) {
    case DType::float32:
      sparse_to_dense<float>(sparse_indices_buf, sparse_values_info.f32(),
                             sparse_values_rank, default_value_info.f32(),
                             slice_rank, num_updates, slice_size, strides,
                             output_size, out_info.f32_write());
      break;
    case DType::int32:
      sparse_to_dense<int32_t>(sparse_indices_buf, sparse_values_info.i32(),
                               sparse_values_rank, default_value_info.i32(),
                               slice_rank, num_updates, slice_size, strides,
                               output_size, out_info.i32_write());
      break;
    case DType::boolean:
      sparse_to_dense<bool>(sparse_indices_buf, sparse_values_info.b(),
                            sparse_values_rank, default_value_info.b(),
                            slice_rank, num_updates, slice_size, strides,
                            output_size, out_info.b_write());
      break;
    default:
      util::warn("SparseToDense for tensor id %d failed. Unknown dtype %d",
                 sparse_indices_id, dtype);
  }
}
}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
