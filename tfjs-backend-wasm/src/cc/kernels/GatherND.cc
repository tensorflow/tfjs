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

#include "src/cc/kernels/GatherND.h"

#include "src/cc/backend.h"
#include "src/cc/util.h"

namespace {

template <typename T>
void gathernd_impl(const T* x_ptr, const int* indices_ptr, size_t num_slices,
                   size_t slice_rank, size_t slice_size,
                   const std::vector<size_t>& strides_ptr, T* out_buf_ptr) {
  for (size_t i = 0; i < num_slices; ++i) {
    size_t flattened_index = 0;

    for (size_t j = 0; j < slice_rank; ++j) {
      flattened_index += *indices_ptr * strides_ptr[j];

      indices_ptr++;
    }

    x_ptr += flattened_index * slice_size;

    for (size_t k = 0; k < slice_size; ++k) {
      *out_buf_ptr += *x_ptr;

      out_buf_ptr++;
      x_ptr++;
    }

    out_buf_ptr += slice_size;
    x_ptr -= ((flattened_index + 1) * slice_size);
  }
}

template void gathernd_impl<float>(const float* x_ptr, const int* indices_ptr,
                                   size_t num_slices, size_t slice_rank,
                                   size_t slice_size,
                                   const std::vector<size_t>& strides_ptr,
                                   float* out_buf_ptr);
template void gathernd_impl<int32_t>(const int* x_ptr, const int* indices_ptr,
                                     size_t num_slices, size_t slice_rank,
                                     size_t slice_size,
                                     const std::vector<size_t>& strides_ptr,
                                     int* out_buf_ptr);
template void gathernd_impl<bool>(const bool* x_ptr, const int* indices_ptr,
                                  size_t num_slices, size_t slice_rank,
                                  size_t slice_size,
                                  const std::vector<size_t>& strides_ptr,
                                  bool* out_buf_ptr);
}  // namespace

namespace tfjs {
namespace wasm {
extern "C" {
#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void GatherND(size_t x_id, const DType dtype, size_t indices_id,
              size_t num_slices, size_t slice_rank, size_t slice_size,
              size_t* strides_ptr, size_t out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& indices_info = backend::get_tensor_info(indices_id);
  const std::vector<size_t>& strides =
      std::vector<size_t>(strides_ptr, strides_ptr + slice_rank);

  const float* x_buf = x_info.f32();
  const int* indices_buf = indices_info.i32();
  auto& out_info = backend::get_tensor_info_out(out_id);
  float* out_buf = out_info.f32_write();

  switch (dtype) {
    case DType::float32:
      gathernd_impl<float>(x_info.f32(), indices_buf, num_slices, slice_rank,
                           slice_size, strides, out_info.f32_write());
      break;
    case DType::int32:
      gathernd_impl<int32_t>(x_info.i32(), indices_buf, num_slices, slice_rank,
                             slice_size, strides, out_info.i32_write());
      break;
    case DType::boolean:
      gathernd_impl<bool>(x_info.b(), indices_buf, num_slices, slice_rank,
                          slice_size, strides, out_info.b_write());
      break;
    default:
      util::warn("Scatter for tensor id %d failed. Unknown dtype %d",
                 indices_id, dtype);
  }
}
}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
