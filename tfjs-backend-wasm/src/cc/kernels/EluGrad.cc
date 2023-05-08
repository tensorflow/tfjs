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

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/elu_impl.h"

namespace tfjs::wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

// REQUIRES
// - Tensor `y`, `dy`, and `out` must have dtype float32 (checked in tfjs-core)
// - Tensor `y` and `dy` must have the same shape (checked in tfjs-core)
void EluGrad(const int y_id, const int dy_id, const int out_id) {
  const size_t y_size = backend::get_tensor_info(y_id).size;
  const float* y_buf = backend::get_tensor_info(y_id).f32();
  const float* dy_buf = backend::get_tensor_info(dy_id).f32();
  float* out_buf = backend::get_tensor_info_out(out_id).f32_write();

  for (size_t i = 0; i < y_size; ++i) {
    if (y_buf[i] >= 0) {
      out_buf[i] = dy_buf[i];
    } else {
      out_buf[i] = dy_buf[i] * (y_buf[i] + 1.0);
    }
  }
}

}  // extern "C"
}  // namespace tfjs::wasm
