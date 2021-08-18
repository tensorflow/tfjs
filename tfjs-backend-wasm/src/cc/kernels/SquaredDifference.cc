/* Copyright 2020 Google LLC. All Rights Reserved.
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
#include "tfjs-backend-wasm/src/cc/unary.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace {
template <class T>
inline T squared_diff(T a, T b) {
  return (a - b) * (a - b);
}
}  // namespace

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void SquaredDifference(
    const size_t a_id, const size_t* a_shape_ptr, const size_t a_shape_len,
    const size_t b_id, const size_t* b_shape_ptr, const size_t b_shape_len,
    const DType dtype, const size_t out_id) {
  switch (dtype) {
    case DType::float32:
      binary_xnn_f32(a_id, a_shape_ptr, a_shape_len, b_id, b_shape_ptr,
                     b_shape_len, out_id, xnn_create_subtract_nd_f32,
                     xnn_setup_subtract_nd_f32);
      unary_xnn_f32(out_id, out_id, xnn_create_square_nc_f32,
                    xnn_setup_square_nc_f32);
      break;
    case DType::int32:
      binary_i32(a_id, b_id, out_id, squared_diff<int32_t>);
      break;
    case DType::boolean:
      binary_bool(a_id, b_id, out_id, squared_diff<bool>);
      break;
    default:
      util::warn("SquaredDifference for tensor ids %d and %d failed. "
                 "Unknown dtype %d", a_id, b_id, dtype);
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
