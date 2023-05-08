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

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace {

template <typename T>
void gather_impl(const T* x_ptr, const std::vector<size_t>& x_strides,
                 const int32_t* indices_ptr, const size_t out_size,
                 const size_t batch_size,
                 const std::vector<size_t>& out_strides, T* out_buf_ptr) {
  for (size_t i = 0; i < out_size; ++i) {
    auto loc = tfjs::util::offset_to_loc(i, out_strides);
    const size_t batch_loc = loc[0];
    const size_t indices_loc = loc[2];
    loc[2] = indices_ptr[batch_loc * batch_size + indices_loc];

    const size_t original_index = tfjs::util::loc_to_offset(loc, x_strides);

    *out_buf_ptr = x_ptr[original_index];

    out_buf_ptr++;
  }
}
}  // namespace

namespace tfjs {
namespace wasm {
extern "C" {
#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void Gather(const size_t x_id, const DType dtype, const int32_t* x_strides_ptr,
            const size_t strides_size, const size_t indices_id,
            const size_t batch_size, const int32_t* out_strides_ptr,
            const size_t out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& indices_info = backend::get_tensor_info(indices_id);

  const int* indices_buf = indices_info.i32();
  auto& out_info = backend::get_tensor_info_out(out_id);
  const size_t out_size = out_info.size;

  const auto x_strides =
      std::vector<size_t>(x_strides_ptr, x_strides_ptr + strides_size);
  const auto out_strides =
      std::vector<size_t>(out_strides_ptr, out_strides_ptr + strides_size);

  switch (dtype) {
    case DType::float32:
      gather_impl<float>(x_info.f32(), x_strides, indices_buf, out_size,
                         batch_size, out_strides, out_info.f32_write());
      break;
    case DType::int32:
      gather_impl<int32_t>(x_info.i32(), x_strides, indices_buf, out_size,
                           batch_size, out_strides, out_info.i32_write());
      break;
    case DType::boolean:
      gather_impl<bool>(x_info.b(), x_strides, indices_buf, out_size,
                        batch_size, out_strides, out_info.b_write());
      break;
    default:
      util::warn("Gather for tensor id %d failed. Unknown dtype %d", x_id,
                 dtype);
  }
}
}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
