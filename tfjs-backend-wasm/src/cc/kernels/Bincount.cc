/* Copyright 2021 Google LLC. All Rights Reserved.
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
#include "tfjs-backend-wasm/src/cc/util.h"

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void Bincount(const size_t x_id, const size_t w_id, const size_t size,
              const size_t out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& w_info = backend::get_tensor_info(w_id);
  auto& out_info = backend::get_tensor_info_out(out_id);

  const int* x_buf = x_info.i32();
  const float* w_buf = w_info.f32();
  float* out_buf = out_info.f32_write();

  const int x_size = x_info.size;
  const int w_size = w_info.size;

  for (size_t i = 0; i < x_size; i++) {
    const int value = x_buf[i];
    if (value < 0) {
      util::warn("Input x must be non-negative!");
    }

    if (value >= size) {
      continue;
    }

    if (w_size > 0) {
      out_buf[value] = out_buf[value] + w_buf[i];
    } else {
      out_buf[value]++;
    }
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
