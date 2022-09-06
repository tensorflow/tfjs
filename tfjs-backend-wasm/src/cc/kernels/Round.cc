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

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/unary.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void Round(const size_t x_id, const DType dtype, const size_t out_id) {
  switch (dtype) {
    case DType::float32: {
      unary_xnn_f32(x_id, out_id, xnn_create_bankers_rounding_nc_f32,
                    xnn_setup_bankers_rounding_nc_f32);
      break;
    }
    case DType::int32:
      unary_i32(x_id, out_id, [](int a) { return a; });
      break;
    default:
      util::warn(
          "Round for tensor ids %d failed. "
          "Unknown dtype %d",
          x_id, dtype);
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
