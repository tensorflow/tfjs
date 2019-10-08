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

#include <emscripten.h>
#include <math.h>
#include <cstdio>
#include <map>
#include <vector>

#include "src/cc/backend.h"
#include "src/cc/util.h"

template <class T>
void add_impl(T* a_buf, int a_size, T* b_buf, int b_size, T* out_buf) {
  int size = std::max(a_size, b_size);
  for (int i = 0; i < size; ++i) {
    out_buf[i] = a_buf[i % a_size] + b_buf[i % b_size];
  }
}
// Templates need explicit instantiation when implemented in a .cc file.
template void add_impl<float>(float* a_buf, int a_size, float* b_buf,
                              int b_size, float* out_buf);
template void add_impl<int>(int* a_buf, int a_size, int* b_buf, int b_size,
                            int* out_buf);
template void add_impl<bool>(bool* a_buf, int a_size, bool* b_buf, int b_size,
                             bool* out_buf);

namespace tfjs {
// We use C-style API to interface with Javascript.
extern "C" {

EMSCRIPTEN_KEEPALIVE
void add(int a_id, int b_id, int out_id) {
  const auto a_info = get_tensor_info(a_id);
  const auto b_info = get_tensor_info(b_id);
  const auto out_info = get_tensor_info(out_id);
  switch (a_info.dtype) {
    case DType::float32:
      add_impl(a_info.buf.f32, a_info.size, b_info.buf.f32, b_info.size,
               out_info.buf.f32);
      break;
    case DType::int32:
      add_impl(a_info.buf.i32, a_info.size, b_info.buf.i32, b_info.size,
               out_info.buf.i32);
      break;
    case DType::boolean:
      add_impl(a_info.buf.b, a_info.size, b_info.buf.b, b_info.size,
               out_info.buf.b);
      break;
    default:
      util::warn("Add for tensor ids %d and %d failed. Unknown dtype %d", a_id,
                 b_id, a_info.dtype);
  }
}

}  // extern "C"
}  // namespace tfjs
