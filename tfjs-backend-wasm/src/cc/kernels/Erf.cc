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

#include "tfjs-backend-wasm/src/cc/unary.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace tfjs {
namespace wasm {

namespace {
template <typename T>
inline T ErfImpl(T n) {
  return static_cast<T>(std::erff(static_cast<float>(n)));
}
}  // namespace

extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void Erf(const int x_id, const DType dtype, const int out_id) {
  switch (dtype) {
    case DType::float32:
      unary_f32(x_id, out_id, ErfImpl<float>);
      break;
    default:
      util::warn("Erf for tensor id %d failed. Unsupported dtype %d", x_id,
                 dtype);
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
