/* Copyright 2023 Google LLC. All Rights Reserved.
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

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace tfjs::wasm {

namespace {

template <typename T>
void ArgMinImpl(const T* x, const int outer_size, const int inner_size,
                int32_t* out_buf) {
  for (int i = 0; i < outer_size; ++i) {
    const int offset = i * inner_size;
    T min_value = x[offset];
    int min_index = 0;
    for (int j = 1; j < inner_size; ++j) {
      T value = x[offset + j];
      if (value < min_value) {
        min_value = value;
        min_index = j;
      }
    }
    out_buf[i] = min_index;
  }
}

}  // namespace

// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void ArgMin(const int x_id, const DType dtype, const int outer_size,
            const int inner_size, const int out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  int32_t* out_buf = out_info.i32_write();

  switch (dtype) {
    case DType::float32:
      ArgMinImpl<float>(x_info.f32(), outer_size, inner_size, out_buf);
      break;
    case DType::int32:
      ArgMinImpl<int32_t>(x_info.i32(), outer_size, inner_size, out_buf);
      break;
    case DType::boolean:
      ArgMinImpl<bool>(x_info.b(), outer_size, inner_size, out_buf);
      break;
    default:
      util::warn("ArgMin failed. Unknown dtype %d", dtype);
  }
}

}  // extern "C"
}  // namespace tfjs::wasm
