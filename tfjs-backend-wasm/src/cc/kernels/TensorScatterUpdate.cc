/* Copyright 2022 Google LLC. All Rights Reserved.
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

#include <cstring>
#include <vector>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace {
template <typename T>
void scatter(const int* indices_ptr, const T* updates_ptr,
             const size_t slice_rank, const size_t num_updates,
             const size_t slice_size, const std::vector<size_t>& strides_ptr,
             const size_t output_size, const size_t dtype_size, T* out_buf_ptr,
             const T* tensor_ptr) {
  // Initialize output to tensor.
  memcpy(out_buf_ptr, tensor_ptr, output_size * dtype_size);

  for (size_t i = 0; i < num_updates; ++i) {
    size_t flattened_index = 0;
    for (size_t j = 0; j < slice_rank; ++j) {
      flattened_index += *indices_ptr * strides_ptr[j];

      indices_ptr++;
    }

    out_buf_ptr += flattened_index * slice_size;

    memcpy(out_buf_ptr, updates_ptr, slice_size * dtype_size);
    out_buf_ptr += slice_size;
    updates_ptr += slice_size;

    out_buf_ptr -= (flattened_index * slice_size + slice_size);
  }
}

}  // namespace

namespace tfjs {
namespace wasm {
extern "C" {
#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void TensorScatterUpdate(const size_t indices_id, const size_t updates_id,
                         const DType dtype, const size_t slice_rank,
                         const size_t num_updates, const size_t slice_size,
                         const size_t* strides_ptr, const size_t output_size,
                         const size_t out_id, const size_t tensor_id) {
  auto& indices_info = backend::get_tensor_info(indices_id);
  auto& updates_info = backend::get_tensor_info(updates_id);
  auto& tensor_info = backend::get_tensor_info(tensor_id);
  const std::vector<size_t>& strides =
      std::vector<size_t>(strides_ptr, strides_ptr + slice_rank);
  const int* indices_buf = indices_info.i32();
  auto& out_info = backend::get_tensor_info_out(out_id);

  switch (dtype) {
    case DType::float32:
      scatter<float>(indices_buf, updates_info.f32(), slice_rank, num_updates,
                     slice_size, strides, output_size, sizeof(float),
                     out_info.f32_write(), tensor_info.f32());
      break;
    case DType::int32:
      scatter<int32_t>(indices_buf, updates_info.i32(), slice_rank, num_updates,
                       slice_size, strides, output_size, sizeof(int32_t),
                       out_info.i32_write(), tensor_info.i32());
      break;
    case DType::boolean:
      scatter<bool>(indices_buf, updates_info.b(), slice_rank, num_updates,
                    slice_size, strides, output_size, sizeof(bool),
                    out_info.b_write(), tensor_info.b());
      break;
    default:
      util::warn("Scatter for tensor id %d failed. Unknown dtype %d",
                 indices_id, dtype);
  }
}
}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
