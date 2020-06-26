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

#include "src/cc/binary.h"
#include "src/cc/util.h"

namespace {
template <class T>
inline bool notEqual(T a, T b) {
  return a != b;
}
}  // namespace

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void NotEqual(const int a_id, const size_t* a_shape_ptr, const int a_shape_len,
              const int b_id, const size_t* b_shape_ptr, const int b_shape_len,
              const DType input_type, const int out_id) {
  switch (input_type) {
    case DType::float32:
      compare_f32(a_id, b_id, out_id, notEqual<float>);
      break;
    case DType::int32:
      compare_i32(a_id, b_id, out_id, notEqual<int>);
      break;
    case DType::boolean:
      compare_bool(a_id, b_id, out_id, notEqual<bool>);
      break;
    default:
      util::warn(
          "NotEqual for tensor ids %d and %d failed."
          "Unsupported input_type %d",
          a_id, b_id, input_type);
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
