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

#include <xnnpack.h>
#include <cmath>
#include <cstddef>
#include <map>
#include <tuple>

#include "src/cc/backend.h"
#include "src/cc/util.h"

namespace tfjs {
namespace wasm {
extern "C" {
#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void LSTM(const size_t i_id, const size_t j_id, const size_t c_id,
          const size_t forget_bias_id, const size_t f_id, const size_t o_id,
          const size_t new_c_id, const size_t new_h_id) {}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
