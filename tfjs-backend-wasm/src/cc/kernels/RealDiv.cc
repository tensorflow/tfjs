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

#include <xnnpack.h>
#include <cstddef>

#include "tfjs-backend-wasm/src/cc/binary.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace {
template <class T>
inline T div(T a, T b) {
  return a / b;
}
}  // namespace

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void RealDiv(const size_t a_id, const size_t* a_shape_ptr,
             const size_t a_shape_len, const size_t b_id,
             const size_t* b_shape_ptr, const size_t b_shape_len,
             const DType dtype, const size_t out_id) {
  switch (dtype) {
    case DType::float32:
      binary_xnn_f32(a_id, a_shape_ptr, a_shape_len, b_id, b_shape_ptr,
                     b_shape_len, out_id, xnn_create_divide_nd_f32,
                     xnn_setup_divide_nd_f32);
      break;
    case DType::int32:
      binary_i32(a_id, a_shape_ptr, a_shape_len, b_id, b_shape_ptr, b_shape_len,
                 out_id, div<int32_t>);
      break;
    case DType::boolean:
      binary_bool(a_id, a_shape_ptr, a_shape_len, b_id, b_shape_ptr,
                  b_shape_len, out_id, div<bool>);
      break;
    default:
      util::warn("Div for tensor ids %d and %d failed. Unknown dtype %d", a_id,
                 b_id, dtype);
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
