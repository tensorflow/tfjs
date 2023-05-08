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

#include <cmath>
#include <cstddef>
#include <iostream>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/leakyrelu_impl.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace {
template <class T>
inline T leakyrelu_op(T a, T b) {
  return a < 0 ? b * a : a;
}
}  // namespace

namespace tfjs {
namespace wasm {

template <typename T>
void leakyrelu(const T* x_buf, const size_t x_size, const float leakyrelu_alpha,
               const size_t out_id) {
  auto& out_info = backend::get_tensor_info_out(out_id);

  float* out_buf = out_info.f32_write();

  for (size_t i = 0; i < out_info.size; i++) {
    float float_x = static_cast<float>(x_buf[i]);
    out_buf[i] = float_x < 0 ? leakyrelu_alpha * float_x : float_x;
  }
}

// Following is required to avoid linker error.
//
// See https://isocpp.org/wiki/faq/templates#separate-template-fn-defn-from-decl

template void leakyrelu<float>(const float* x_buf, const size_t x_size,
                               const float leakyrelu_alpha,
                               const size_t out_id);

template void leakyrelu<int32_t>(const int32_t* x_buf, const size_t x_size,
                                 const float leakyrelu_alpha,
                                 const size_t out_id);

}  // namespace wasm
}  // namespace tfjs
