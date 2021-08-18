
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

#include "tfjs-backend-wasm/src/cc/backend.h"

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void SelectV2(const int condition_id, const int t_id, const int e_id,
              const int offset, const int out_id) {
  auto& condition_info = backend::get_tensor_info(condition_id);
  auto& t_info = backend::get_tensor_info(t_id);
  auto& e_info = backend::get_tensor_info(e_id);
  auto& out_info = backend::get_tensor_info_out(out_id);

  const bool* condition_buf = condition_info.b();
  const float* t_buf = t_info.f32();
  const float* e_buf = e_info.f32();
  float* out_buf = out_info.f32_write();

  const size_t values_size = condition_info.size;

  for (size_t i = 0; i < values_size; ++i) {
    for (size_t j = 0; j < offset; ++j) {
      if (condition_buf[i]) {
        *out_buf = t_buf[i];
      } else {
        *out_buf = e_buf[i];
      }
      out_buf++;
    }
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
