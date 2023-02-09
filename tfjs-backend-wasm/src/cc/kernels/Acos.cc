/* Copyright 2023 Google LLC.
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

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/unary.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace {

template <typename T>
inline T acos_impl(T n) {
  return static_cast<T>(std::acosf(static_cast<float>(n)));
}

}  // namespace

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void Acos(const int x_id, const DType dtype, const int out_id) {
  switch (dtype) {
    case DType::float32:
      unary_f32(x_id, out_id, acos_impl<float>);
      break;
    case DType::int32:
      unary_i32(x_id, out_id, acos_impl<int>);
      break;
    default:
      util::warn("Acos for tensor id %d failed. Unsupported dtype %d", x_id,
                 dtype);
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
