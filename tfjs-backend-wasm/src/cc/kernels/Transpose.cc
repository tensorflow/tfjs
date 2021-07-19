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

#include <cstddef>
#include <vector>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/transpose_impl.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void Transpose(const size_t x_id, const size_t* x_shape_ptr,
               const size_t x_shape_length, const DType dtype,
               const size_t out_id, size_t* perm_ptr,
               const size_t perm_length) {
  auto x_shape = std::vector<size_t>(x_shape_ptr, x_shape_ptr + x_shape_length);
  auto perm = std::vector<size_t>(perm_ptr, perm_ptr + perm_length);
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info_out(out_id);

  switch (dtype) {
    case DType::float32:
      tfjs::wasm::transpose<float>(x_info.f32(), x_shape, perm,
                                   out_info.f32_write());
      break;
    case DType::int32:
      tfjs::wasm::transpose<int32_t>(x_info.i32(), x_shape, perm,
                                     out_info.i32_write());
      break;
    case DType::boolean:
      tfjs::wasm::transpose<bool>(x_info.b(), x_shape, perm,
                                  out_info.b_write());
      break;
    default:
      util::warn("Transpose for tensor id %d failed. Unknown dtype %d", x_id,
                 dtype);
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
