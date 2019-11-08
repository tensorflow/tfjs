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

#include "src/cc/backend.h"
#include "src/cc/binary.h"
#include "src/cc/util.h"

namespace {
template <class T>
inline T add(T a, T b) {
  return a + b;
}
}  // namespace

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void Add(const int a_id, const int b_id, const DType dtype, const int out_id) {
  auto& a_info = backend::get_tensor_info(a_id);

  switch (dtype) {
    case DType::float32:
      binary_f32(a_id, b_id, out_id, add<float>);
      break;
    case DType::int32:
      binary_i32(a_id, b_id, out_id, add<int>);
      break;
    case DType::boolean:
      binary_bool(a_id, b_id, out_id, add<bool>);
      break;
    default:
      tfjs::backend::throw_js_exception(
          "Add for tensor ids %d and %d failed. Unknown dtype %d", a_id, b_id,
          dtype);
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
