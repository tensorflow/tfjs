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

#include <cstddef>

#include "tfjs-backend-wasm/src/cc/binary.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace {
template <class T>
inline bool equal(T a, T b) {
  return a == b;
}
}  // namespace

namespace tfjs {
namespace wasm {
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void Equal(const size_t a_id, const size_t* a_shape_ptr,
           const size_t a_shape_len, const size_t b_id,
           const size_t* b_shape_ptr, const size_t b_shape_len,
           const DType input_type, const size_t out_id) {
  switch (input_type) {
    case DType::float32:
      compare_f32(a_id, b_id, out_id, equal<float>);
      break;
    case DType::int32:
      compare_i32(a_id, b_id, out_id, equal<int>);
      break;
    case DType::boolean:
      compare_bool(a_id, b_id, out_id, equal<bool>);
      break;
    default:
      util::warn(
          "Equal for tensor ids %d and %d failed. Unsupported input_type %d",
          a_id, b_id, input_type);
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
