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

#include <cmath>
#include <cstddef>

#include "src/cc/binary.h"
#include "src/cc/util.h"

namespace {
template <class T>
inline T power(T a, T b) {
  return pow(a, b);
}
}  // namespace

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void Pow(const size_t a_id, const size_t* a_shape_ptr, const size_t a_shape_len,
         const size_t b_id, const size_t* b_shape_ptr, const size_t b_shape_len,
         const DType dtype, const size_t out_id) {
  switch (dtype) {
    case DType::float32:
      binary_f32(a_id, b_id, out_id, power<float>);
      break;
    case DType::int32:
      binary_i32(a_id, b_id, out_id, power<int32_t>);
      break;
    case DType::boolean:
      binary_bool(a_id, b_id, out_id, power<bool>);
      break;
    default:
      util::warn("Pow for tensor ids %d and %d failed. Unknown dtype %d", a_id,
                 b_id, dtype);
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
