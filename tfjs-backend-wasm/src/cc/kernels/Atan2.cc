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

#include <cmath>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/binary.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace tfjs::wasm {

namespace {

template <typename T>
inline T Atan2Impl(T a, T b) {
  return static_cast<T>(
      std::atan2f(static_cast<float>(a), static_cast<float>(b)));
}

}  // namespace

// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void Atan2(const int a_id, const size_t* a_shape_ptr, const int a_shape_len,
           const int b_id, const size_t* b_shape_ptr, const int b_shape_len,
           const DType dtype, const int out_id) {
  switch (dtype) {
    case DType::float32:
      binary_f32(a_id, a_shape_ptr, a_shape_len, b_id, b_shape_ptr, b_shape_len,
                 out_id, Atan2Impl<float>);
      break;
    case DType::int32:
      binary_i32(a_id, a_shape_ptr, a_shape_len, b_id, b_shape_ptr, b_shape_len,
                 out_id, Atan2Impl<int32_t>);
      break;
    default:
      util::warn("Atan2 for tensor id %d failed. Unsupported dtype %d", a_id,
                 dtype);
  }
}

}  // extern "C"
}  // namespace tfjs::wasm
