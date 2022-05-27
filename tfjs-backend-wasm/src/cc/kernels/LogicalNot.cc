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

#include "tfjs-backend-wasm/src/cc/binary.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace {
inline bool logical_not(bool a) { return !a; }
} // namespace

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void LogicalNot(const int a_id, const size_t *a_shape_ptr,
                const int a_shape_len, const DType input_type,
                const int out_id) {
  switch (input_type) {
  case DType::boolean:
    compare_not(a_id, a_shape_ptr, a_shape_len, out_id, logical_not);
    break;
  default:
    util::warn("LogicalNot for tensor ids %d failed. Unsupported input_type "
               "%d",
               a_id, input_type);
  }
}

} // extern "C"
} // namespace wasm
} // namespace tfjs
