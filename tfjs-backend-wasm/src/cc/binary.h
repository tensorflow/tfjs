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

#ifndef BINARY_H_
#define BINARY_H_

#include <xnnpack.h>
#include <algorithm>

#include "src/cc/backend.h"

namespace tfjs {
namespace wasm {

template <class T>
inline void binary_impl(const T* a_buf, const int a_size, const T* b_buf,
                        const int b_size, T* out_buf, T operation(T, T)) {
  int size = std::max(a_size, b_size);
  for (int i = 0; i < size; ++i) {
    out_buf[i] = operation(a_buf[i % a_size], b_buf[i % b_size]);
  }
}

inline void binary_f32(const int a_id, const int b_id, const int out_id,
                       float operation(float, float)) {
  auto& a_info = backend::get_tensor_info(a_id);
  auto& b_info = backend::get_tensor_info(b_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  binary_impl<float>(a_info.f32(), a_info.size, b_info.f32(), b_info.size,
                     out_info.f32_write(), operation);
}

inline void binary_i32(const int a_id, const int b_id, const int out_id,
                       int operation(int, int)) {
  auto& a_info = backend::get_tensor_info(a_id);
  auto& b_info = backend::get_tensor_info(b_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  binary_impl<int>(a_info.i32(), a_info.size, b_info.i32(), b_info.size,
                   out_info.i32_write(), operation);
}

inline void binary_bool(const int a_id, const int b_id, const int out_id,
                        bool operation(bool, bool)) {
  auto& a_info = backend::get_tensor_info(a_id);
  auto& b_info = backend::get_tensor_info(b_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  binary_impl<bool>(a_info.b(), a_info.size, b_info.b(), b_info.size,
                    out_info.b_write(), operation);
}

typedef xnn_status (*xnn_create_binary_op)(float, float, uint32_t,
                                           xnn_operator_t*);
typedef xnn_status (*xnn_setup_binary_op)(xnn_operator_t, size_t, const size_t*,
                                          size_t, const size_t*, const float*,
                                          const float*, float*, pthreadpool_t);

void binary_xnn_f32(const int a_id, const size_t* a_shape_ptr,
                    const int a_shape_len, const int b_id,
                    const size_t* b_shape_ptr, const int b_shape_len,
                    const int out_id, xnn_create_binary_op create_op,
                    xnn_setup_binary_op setup_op);

}  // namespace wasm
}  // namespace tfjs

#endif  // BINARY_H_
