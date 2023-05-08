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
#include <math.h>

#include <cstddef>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace {

template <typename T>
void max(const T* x_buf, const size_t out_size, const size_t reduce_size,
         T* out_buf) {
  const T* x_offset = x_buf;

  for (size_t i = 0; i < out_size; ++i) {
    const size_t offset = i * reduce_size;
    T max = x_buf[offset];

    const T* x_iter_end = x_offset + reduce_size;

    for (const T* x = x_offset; x < x_iter_end; ++x) {
      T value = *x;
      if (isnan(value) || value > max) {
        max = value;
      }
    }

    x_offset += reduce_size;

    out_buf[i] = max;
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
void Max(const size_t x_id, const DType dtype, const size_t reduce_size,
         const size_t out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info_out(out_id);

  const size_t out_size = out_info.size;

  switch (dtype) {
    case DType::float32:
      max<float>(x_info.f32(), out_size, reduce_size, out_info.f32_write());
      break;
    case DType::int32:
      max<int32_t>(x_info.i32(), out_size, reduce_size, out_info.i32_write());
      break;
    case DType::boolean:
      max<bool>(x_info.b(), out_size, reduce_size, out_info.b_write());
      break;
    default:
      util::warn("Max failed. Unknown dtype %d", dtype);
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
