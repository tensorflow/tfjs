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
#include "tfjs-backend-wasm/src/cc/elu_impl.h"

namespace tfjs::wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

// REQUIRES
// - Tensor `x` and `out` must have dtype float32 (checked in tfjs-core)
void Elu(const int x_id, DType, const int out_id) {
  const TensorInfo& x_info = backend::get_tensor_info(x_id);
  TensorInfo& out_info = backend::get_tensor_info_out(out_id);

  tfjs::wasm::EluImpl(x_info.f32(), x_info.size, out_info.f32_write());
}

}  // extern "C"
}  // namespace tfjs::wasm
