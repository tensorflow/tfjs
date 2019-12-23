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

#include "src/cc/kernels/Gather.h"

#include "src/cc/backend.h"
#include "src/cc/util.h"

namespace {

template <typename T>
void gather_impl(const T* x_ptr, const int* indices_ptr, const size_t axis,
                 const size_t out_size, T* out_buf_ptr) {
  for (size_t i = 0; i < out_size; ++i) {
    *out_buf_ptr = i;

    out_buf_ptr++;
  }
}

template void gather_impl<float>(const float* x_ptr, const int* indices_ptr,
                                 const size_t axis, const size_t out_size,
                                 float* out_buf);
template void gather_impl<int32_t>(const int* x_ptr, const int* indices_ptr,
                                   const size_t axis, const size_t out_size,
                                   int* out_buf);
template void gather_impl<bool>(const bool* x_ptr, const int* indices_ptr,
                                const size_t axis, const size_t out_size,
                                bool* out_buf);
}  // namespace

namespace tfjs {
namespace wasm {
extern "C" {
#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void Gather(size_t x_id, const DType dtype, size_t indices_id, size_t axis,
            size_t out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& indices_info = backend::get_tensor_info(indices_id);

  const int* indices_buf = indices_info.i32();
  auto& out_info = backend::get_tensor_info_out(out_id);
  const size_t out_size = out_info.size;

  switch (dtype) {
    case DType::float32:
      gather_impl<float>(x_info.f32(), indices_buf, axis, out_size,
                         out_info.f32_write());
      break;
    case DType::int32:
      gather_impl<int32_t>(x_info.i32(), indices_buf, axis, out_size,
                           out_info.i32_write());
      break;
    case DType::boolean:
      gather_impl<bool>(x_info.b(), indices_buf, axis, out_size,
                        out_info.b_write());
      break;
    default:
      util::warn("Scatter for tensor id %d failed. Unknown dtype %d", x_id,
                 dtype);
  }
}
}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
