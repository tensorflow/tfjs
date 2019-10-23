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

#include "src/cc/backend.h"

namespace {
template <class T>
inline void binary_impl(T* a_buf, int a_size, T* b_buf, int b_size, T* out_buf,
                        T operation(T, T)) {
  int size = std::max(a_size, b_size);
  for (int i = 0; i < size; ++i) {
    out_buf[i] = operation(a_buf[i % a_size], b_buf[i % b_size]);
  }
}
}  // namespace

namespace tfjs {
namespace wasm {

inline void binary_f32(int a_id, int b_id, int out_id,
                       float operation(float, float)) {
  const auto a_info = backend::get_tensor_info(a_id);
  const auto b_info = backend::get_tensor_info(b_id);
  const auto out_info = backend::get_tensor_info(out_id);
  binary_impl<float>(a_info.buf.f32, a_info.size, b_info.buf.f32, b_info.size,
                     out_info.buf.f32, operation);
}

inline void binary_i32(int a_id, int b_id, int out_id,
                       int operation(int, int)) {
  const auto a_info = backend::get_tensor_info(a_id);
  const auto b_info = backend::get_tensor_info(b_id);
  const auto out_info = backend::get_tensor_info(out_id);
  binary_impl<int>(a_info.buf.i32, a_info.size, b_info.buf.i32, b_info.size,
                   out_info.buf.i32, operation);
}

inline void binary_bool(int a_id, int b_id, int out_id,
                        bool operation(bool, bool)) {
  const auto a_info = backend::get_tensor_info(a_id);
  const auto b_info = backend::get_tensor_info(b_id);
  const auto out_info = backend::get_tensor_info(out_id);
  binary_impl<bool>(a_info.buf.b, a_info.size, b_info.buf.b, b_info.size,
                    out_info.buf.b, operation);
}

}  // namespace wasm
}  // namespace tfjs
