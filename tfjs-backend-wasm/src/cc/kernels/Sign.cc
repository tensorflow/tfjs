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
#include <limits>

#include "tfjs-backend-wasm/src/cc/unary.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace tfjs::wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void Sign(const int x_id, const DType dtype, const int out_id) {
  switch (dtype) {
    case DType::float32:
      unary_f32(x_id, out_id, [](float n) -> float {
        static constexpr float kEps = std::numeric_limits<float>::epsilon();
        if (std::isnan(n)) return 0;
        if (n < kEps && n > -kEps) return 0;
        return n > 0 ? 1 : -1;
      });
      break;
    case DType::int32:
      unary_i32(x_id, out_id, [](int32_t n) {
        if (n == 0) return 0;
        return n > 0 ? 1 : -1;
      });
      break;
    default:
      util::warn("Sign for tensor id %d failed. Unsupported dtype %d", x_id,
                 dtype);
  }
}

}  // extern "C"
}  // namespace tfjs::wasm
