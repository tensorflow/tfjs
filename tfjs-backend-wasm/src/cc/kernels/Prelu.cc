/* Copyright 2019 Google LLC. All Rights Reserved.
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

#include "src/cc/kernels/Prelu.h"

#include <xnnpack.h>
#include <cmath>
#include <cstddef>
#include <unordered_map>

#include "src/cc/backend.h"
#include "src/cc/prelu_impl.h"
#include "src/cc/util.h"

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void Prelu(const size_t x_id, const size_t weights_id, const size_t out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  const float* x_buf = x_info.f32();

  tfjs::wasm::prelu(x_buf, x_info.size, weights_id, out_id);
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
