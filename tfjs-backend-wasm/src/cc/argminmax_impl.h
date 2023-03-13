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

#ifndef ARGMINMAX_IMPL_H_
#define ARGMINMAX_IMPL_H_

#include <cstddef>
#include "tfjs-backend-wasm/src/cc/backend.h"

namespace tfjs::wasm {

namespace {

template <typename T, typename F>
inline void ArgMinMaxInner(const T* x, const size_t outer_size,
                           const size_t inner_size, int32_t* out_buf,
                           const F& update_cond) {
  for (int i = 0; i < outer_size; ++i) {
    const int offset = i * inner_size;
    T target_value = x[offset];
    int target_index = 0;
    for (int j = 1; j < inner_size; ++j) {
      T value = x[offset + j];
      if (update_cond(target_value, value)) {
        target_value = value;
        target_index = j;
      }
    }
    out_buf[i] = target_index;
  }
}

}  // namespace

template <typename T>
inline void ArgMaxImpl(const T* x, const size_t outer_size,
                       const size_t inner_size, int32_t* out_buf) {
  ArgMinMaxInner(
      x, outer_size, inner_size, out_buf,
      [](const T& target, const T& current) { return target < current; });
}

template <typename T>
inline void ArgMinImpl(const T* x, const size_t outer_size,
                       const size_t inner_size, int32_t* out_buf) {
  ArgMinMaxInner(
      x, outer_size, inner_size, out_buf,
      [](const T& target, const T& current) { return target > current; });
}

}  // namespace tfjs::wasm

#endif
