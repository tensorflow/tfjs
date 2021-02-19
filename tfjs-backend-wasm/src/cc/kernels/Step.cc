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

#include <math.h>

#include "tfjs-backend-wasm/src/cc/backend.h"

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void Step(const int x_id, const float alpha, const int out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info_out(out_id);

  const float* x_buf = x_info.f32();
  float* out_buf = out_info.f32_write();
  for (size_t i = 0; i < x_info.size; ++i) {
    if (isnan(x_buf[i])) {
      out_buf[i] = x_buf[i];
    } else {
      out_buf[i] = x_buf[i] > 0 ? 1 : alpha;
    }
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
