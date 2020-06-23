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

#include <cmath>
#include <vector>

#include "src/cc/backend.h"
#include "src/cc/fft_impl.h"

namespace tfjs {
namespace wasm {
extern "C" {
#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void FFT(const size_t real_input_id, const size_t imag_input_id,
         const size_t outer_dim, const size_t inner_dim,
         const size_t is_real_component, const size_t out_id) {
  const bool is_inverse = false;
  tfjs::wasm::fft(real_input_id, imag_input_id, outer_dim, inner_dim,
                  is_real_component, is_inverse, out_id);
}
}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
