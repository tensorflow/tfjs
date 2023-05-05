/* Copyright 2023 Google LLC.
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
#include <cstddef>

#include "tfjs-backend-wasm/src/cc/binary.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace {
template <typename T>
inline T BitwiseAndImp(T a, T b) {
  return a & b;
}
}  // namespace

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {
#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
// REQUIRES:
// - Tensor `a` and `b` must have dtype int32 (checked in tfjs-core)
// - Tensor `a` and `b` must have the same shape (checked in tfjs-core)
void BitwiseAnd(const size_t a_id, const size_t* a_shape_ptr,
                const size_t a_shape_len, const size_t b_id,
                const size_t* b_shape_ptr, const size_t b_shape_len,
                const DType dtype, const size_t out_id) {
  binary_i32(a_id, a_shape_ptr, a_shape_len, b_id, b_shape_ptr, b_shape_len,
             out_id,
             BitwiseAndImp<int32_t>);  // input numbers are ensured to be int32
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
