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

#include <algorithm>
#include <cmath>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace tfjs {
namespace wasm {

namespace {

template <typename T>
inline void DiagImpl(const T* x_buf, int32_t x_size, T* out_buf) {
  std::fill(out_buf, out_buf + x_size * x_size, 0);
  for (int32_t i = 0; i < x_size; ++i) {
    out_buf[x_size * i + i] = x_buf[i];
  }
}

}  // namespace

// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void Diag(const int32_t x_id, const DType dtype, const int32_t x_size,
          const int32_t out_id) {
  const TensorInfo& x_info = backend::get_tensor_info(x_id);
  TensorInfo& out_info = backend::get_tensor_info_out(out_id);
  switch (dtype) {
    case DType::float32:
      DiagImpl(x_info.f32(), x_size, out_info.f32_write());
      break;
    case DType::int32:
      DiagImpl(x_info.i32(), x_size, out_info.i32_write());
      break;
    case DType::boolean:
      DiagImpl(x_info.b(), x_size, out_info.b_write());
      break;
    default:
      util::warn("Diag for tensor id %d failed. Unsupported dtype %d", x_id,
                 dtype);
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
