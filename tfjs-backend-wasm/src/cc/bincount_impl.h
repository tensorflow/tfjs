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

#ifndef BINCOUNT_IMPL_H_
#define BINCOUNT_IMPL_H_

#include <cstdint>

#include "tfjs-backend-wasm/src/cc/util.h"

namespace tfjs::wasm {
// REQUIRES:
// weight_buf must have the same size as x_buf.
template <bool reset_out_buf = true, typename T>
inline void BincountImpl(const int32_t* x_buf, int32_t x_len, int32_t size,
                         const T* weight_buf, bool binary_output, T* out_buf) {
  if (reset_out_buf) {
    std::fill(out_buf, out_buf + size, 0);
  }
  for (int32_t i = 0; i < x_len; ++i) {
    int32_t value = x_buf[i];
    if (value < 0) {
      util::warn("DenseBincount error: input x must be non-negative.");
      continue;
    }
    if (value >= size) {
      continue;
    }

    if (binary_output) {
      out_buf[value] = 1;
    } else if (weight_buf == nullptr) {
      out_buf[value] += 1;
    } else {
      out_buf[value] += weight_buf[i];
    }
  }
}
}  // namespace tfjs::wasm
#endif
