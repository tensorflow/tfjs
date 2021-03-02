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
#include <cstdint>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace tfjs {

template <typename T>
void cumsum(const size_t x_id, const size_t exclusive, const size_t reverse,
            const size_t final_dim, const size_t out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info_out(out_id);

  const T* x_buf = reinterpret_cast<const T*>(x_info.memory_offset);
  T* out_buf = reinterpret_cast<T*>(out_info.memory_offset);

  for (size_t i = 0; i < x_info.size; i += final_dim) {
    for (size_t j = 0; j < final_dim; ++j) {
      const size_t idx = reverse ? i + final_dim - j - 1 : i + j;
      if (j == 0) {
        out_buf[idx] = exclusive ? 0 : x_buf[idx];
      } else {
        const size_t prev_idx = reverse ? idx + 1 : idx - 1;
        out_buf[idx] = exclusive ? x_buf[prev_idx] + out_buf[prev_idx] :
                                   x_buf[idx] + out_buf[prev_idx];
      }
    }
  }
}

namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void Cumsum(const size_t x_id, const size_t exclusive, const size_t reverse,
            const size_t final_dim, const size_t out_id, const DType dtype) {
  switch (dtype) {
    case DType::float32:
      cumsum<float>(x_id, exclusive, reverse, final_dim, out_id);
      break;
    case DType::int32:
      cumsum<int32_t>(x_id, exclusive, reverse, final_dim, out_id);
      break;
    default:
      util::warn("Cumsum for tensor id %d failed. Unsupported dtype %d",
                 x_id, dtype);
  }
}

}   // extern "C"
}   // namespace wasm
}   // namespace tfjs
