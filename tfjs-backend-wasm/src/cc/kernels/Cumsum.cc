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

#include <cstddef>
#include <vector>

#include "src/cc/backend.h"
#include "src/cc/util.h"

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void Cumsum(const size_t x_id, const size_t exclusive, const size_t reverse,
            const size_t final_dim, const size_t out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info_out(out_id);

  const float* x_buf = x_info.f32();
  float* out_buf = out_info.f32_write();

  size_t x_size = x_info.size;

  for (size_t i = 0; i < x_size; ++i) {
    for (size_t j = 0; j < final_dim; ++j) {
      size_t idx;
      if (reverse > 0) {
        idx = i + final_dim - j - 1;
      } else {
        idx = i + j;
      }

      if (j == 0) {
        if (exclusive == 0) {
          *out_buf = x_buf[idx];
        } else {
          *out_buf = 0;
        }
      } else {
        size_t prev_idx;
        if (reverse > 0) {
          prev_idx = i + final_dim - j - 2;
        } else {
          prev_idx = i + j - 1;
        }

        if (exclusive > 0) {
          *out_buf = x_buf[prev_idx] + out_buf[prev_idx];
        } else {
          *out_buf = x_buf[idx] + out_buf[prev_idx];
        }
      }

      out_buf++;
    }
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
