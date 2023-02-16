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

namespace {

template <typename T>
inline T SeluImpl(T n) {
  static constexpr float kScaleAlpha = 1.7580993408473768599402175208123;
  static constexpr float kScale = 1.0507009873554804934193349852946;

  if (n > 0) return kScale * n;
  return kScaleAlpha * (std::expf(static_cast<float>(n)) - 1.0);
}

}  // namespace

namespace tfjs::wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void Selu(const int x_id, const DType dtype, const int out_id) {
  switch (dtype) {
    case DType::float32:
      unary_f32(x_id, out_id, SeluImpl<float>);
      break;
    case DType::int32:
      unary_i32(x_id, out_id, SeluImpl<int32_t>);
      break;
    default:
      util::warn("Selu for tensor id %d failed. Unsupported dtype %d", x_id,
                 dtype);
  }
}

}  // extern "C"
}  // namespace tfjs::wasm
