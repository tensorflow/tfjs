/* Copyright 2020 Google LLC. All Rights Reserved.
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
#include <cstddef>

#include "src/cc/backend.h"

namespace tfjs {
namespace wasm {
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void OneHot(const size_t indices_id, const size_t depth, const int32_t on_value,
            const int32_t off_value, const size_t out_id) {
  auto& indices_info = backend::get_tensor_info(indices_id);
  auto& out_info = backend::get_tensor_info_out(out_id);

  const int* indices_buf = indices_info.i32();
  const size_t indices_size = indices_info.size;

  int* out_buf = out_info.i32_write();
  const size_t out_size = out_info.size;

  // Initialize output with off_value.
  std::fill(out_buf, out_buf + out_size, off_value);

  for (size_t i = 0; i < indices_size; ++i) {
    if (indices_buf[i] >= 0 && indices_buf[i] < depth) {
      out_buf[i * depth + indices_buf[i]] = on_value;
    }
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
