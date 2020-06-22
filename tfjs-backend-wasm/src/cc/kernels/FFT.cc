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

#include <vector>

#include "src/cc/backend.h"

namespace tfjs {
namespace wasm {
extern "C" {
#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void FFT(const size_t input_id, const size_t out_id) {
  auto& input_info = backend::get_tensor_info(input_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  const float* input_buf = input_info.f32();
  float* out_buf = out_info.f32_write();
  const size_t input_size = input_info.size;

  for (size_t i = 0; i < input_size; ++i) {
    out_buf[i] = input_buf[i];
  }
}
}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
