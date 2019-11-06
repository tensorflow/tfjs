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

#include "src/cc/backend.h"

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void Min(const int x_id, const int reduce_size, const int out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info(out_id);

  const float* x_buf = reinterpret_cast<float*>(x_info.memory_offset);
  const int x_size = x_info.size;

  float* out_buf = reinterpret_cast<float*>(out_info.memory_offset);
  const int out_size = out_info.size;

  float* x_offset = const_cast<float*>(x_buf);

  for (int i = 0; i < out_size; ++i) {
    const int offset = i * reduce_size;
    float min = x_buf[offset];

    const float* x_iter_end = x_offset + reduce_size;

    for (float* x = x_offset; x < x_iter_end; ++x) {
      float value = *x;
      if (value < min) {
        min = value;
      }
    }

    x_offset += reduce_size;

    out_buf[i] = min;
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
