/* Copyright 2023 Google LLC. All Rights Reserved.
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

namespace tfjs::wasm {

extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

// REQUIRES:
// - Tensor `out` must have dtype float32.
// - Param `num` must be positive (checked in tfjs-core).
void LinSpace(int32_t out_id, float start, float stop, int32_t num) {
  TensorInfo& out_info = backend::get_tensor_info_out(out_id);
  float* out_buf = out_info.f32_write();

  out_buf[0] = start;
  if (num == 1) {
    return;
  }

  float step = (stop - start) / static_cast<float>(num - 1);
  for (int i = 1; i < num; ++i) {
    out_buf[i] = out_buf[i - 1] + step;
  }
}

}  // extern "C"

}  // namespace tfjs::wasm
