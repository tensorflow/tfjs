/* Copyright 2019 Google Inc. All Rights Reserved.
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
#include "src/cc/util.h"

namespace {

template <typename T>
void argmax(const T* x, const size_t outer_size, const size_t inner_size,
            size_t* out_buf) {
  for (size_t i = 0; i < outer_size; ++i) {
    const size_t offset = i * inner_size;
    T max = x[offset];
    size_t max_index = 0;
    for (size_t j = 1; j < inner_size; ++j) {
      const T val = x[offset + j];
      if (val > max) {
        max = val;
        max_index = j;
      }
    }
    out_buf[i] = max_index;
  }
}

}  // namespace

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void ArgMax(const size_t x_id, const DType dtype, const size_t outer_size,
            const size_t inner_size, const size_t out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  size_t* out_buf = out_info.i32_write();

  switch (dtype) {
    case DType::float32:
      argmax<float>(x_info.f32(), outer_size, inner_size, out_buf);
      break;
    case DType::int32:
      argmax<size_t>(x_info.i32(), outer_size, inner_size, out_buf);
      break;
    case DType::boolean:
      argmax<bool>(x_info.b(), outer_size, inner_size, out_buf);
      break;
    default:
      util::warn("Argmax failed. Unknown dtype %d", dtype);
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
