/* Copyright 2019 Google Inc. All Rights Reserved.
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
#include <vector>
#include "src/cc/backend.h"

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void Max(int x_id, int reduce_size, int out_id) {
  const auto x_info = backend::get_tensor_info(x_id);
  const auto out_info = backend::get_tensor_info(out_id);

  float* x_buf = x_info.buf.f32;
  int x_size = x_info.size;

  float* out_buf = out_info.buf.f32;
  int out_size = out_info.size;

  for (int i = 0; i < out_size; ++i) {
    int offset = i * reduce_size;
    float max = x_buf[offset];
    for (int j = 0; j < reduce_size; ++j) {
      float value = x_buf[offset + j];
      if (value > max) {
        max = value;
      }
    }
    out_buf[i] = max;
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
