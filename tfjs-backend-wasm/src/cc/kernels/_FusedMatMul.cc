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

#include <cstddef>

#include "src/cc/backend.h"
#include "src/cc/batch_mat_mul_impl.h"
#include "src/cc/kernels/_FusedMatMul.h"

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void _FusedMatMul(const size_t a_id, const size_t* a_shape_ptr,
                  const size_t a_shape_len, const size_t b_id,
                  const size_t* b_shape_ptr, const size_t b_shape_len,
                  const bool transpose_a, const bool transpose_b,
                  const FusableActivation activation, const size_t bias_id,
                  const size_t prelu_weights_id, const size_t out_id) {
  tfjs::wasm::fused_batch_mat_mul(
      a_id, a_shape_ptr, a_shape_len, b_id, b_shape_ptr, b_shape_len,
      transpose_a, transpose_b, activation, bias_id, prelu_weights_id, out_id);
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
