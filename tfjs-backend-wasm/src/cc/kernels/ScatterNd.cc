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

#include <cstring>
#include <vector>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/scatter_impl.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace {
template <typename T>
void scatter_nd(const int* indices_ptr, const T* updates_ptr,
                const size_t slice_rank, const size_t num_updates,
                const size_t slice_size, const std::vector<size_t>& strides_ptr,
                const size_t output_size, T* out_buf_ptr) {
  T default_value{};
  bool sum_dupe_indices = true;
  bool update_as_scalar = false;
  tfjs::wasm::scatter(indices_ptr, updates_ptr, slice_rank, num_updates,
                      slice_size, strides_ptr, output_size, default_value,
                      sum_dupe_indices, update_as_scalar, out_buf_ptr);
}

}  // namespace

namespace tfjs {
namespace wasm {
extern "C" {
#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void ScatterNd(const size_t indices_id, const size_t updates_id,
               const DType dtype, const size_t slice_rank,
               const size_t num_updates, const size_t slice_size,
               const size_t* strides_ptr, const size_t output_size,
               const size_t out_id) {
  auto& indices_info = backend::get_tensor_info(indices_id);
  auto& updates_info = backend::get_tensor_info(updates_id);
  const std::vector<size_t>& strides =
      std::vector<size_t>(strides_ptr, strides_ptr + slice_rank);
  const int* indices_buf = indices_info.i32();
  auto& out_info = backend::get_tensor_info_out(out_id);

  switch (dtype) {
    case DType::float32:
      scatter_nd<float>(indices_buf, updates_info.f32(), slice_rank,
                        num_updates, slice_size, strides, output_size,
                        out_info.f32_write());
      break;
    case DType::int32:
      scatter_nd<int32_t>(indices_buf, updates_info.i32(), slice_rank,
                          num_updates, slice_size, strides, output_size,
                          out_info.i32_write());
      break;
    case DType::boolean:
      scatter_nd<bool>(indices_buf, updates_info.b(), slice_rank, num_updates,
                       slice_size, strides, output_size, out_info.b_write());
      break;
    default:
      util::warn("Scatter for tensor id %d failed. Unknown dtype %d",
                 indices_id, dtype);
  }
}
}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
