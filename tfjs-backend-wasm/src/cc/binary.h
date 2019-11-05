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
  auto& out_info = backend::get_tensor_info(out_id);
  binary_impl<float>(
      reinterpret_cast<const float*>(a_info.memory_offset), a_info.size,
      reinterpret_cast<const float*>(b_info.memory_offset), b_info.size,
      reinterpret_cast<float*>(out_info.memory_offset), operation);
}

inline void binary_i32(const int a_id, const int b_id, const int out_id,
                       int operation(int, int)) {
  auto& a_info = backend::get_tensor_info(a_id);
  auto& b_info = backend::get_tensor_info(b_id);
  auto& out_info = backend::get_tensor_info(out_id);
  binary_impl<int>(
      reinterpret_cast<const int*>(a_info.memory_offset), a_info.size,
      reinterpret_cast<const int*>(b_info.memory_offset), b_info.size,
      reinterpret_cast<int*>(out_info.memory_offset), operation);
}

inline void binary_bool(const int a_id, const int b_id, const int out_id,
                        bool operation(bool, bool)) {
  auto& a_info = backend::get_tensor_info(a_id);
  auto& b_info = backend::get_tensor_info(b_id);
  auto& out_info = backend::get_tensor_info(out_id);
  binary_impl<bool>(
      reinterpret_cast<const bool*>(a_info.memory_offset), a_info.size,
      reinterpret_cast<const bool*>(b_info.memory_offset), b_info.size,
      reinterpret_cast<bool*>(out_info.memory_offset), operation);
}

}  // namespace wasm
}  // namespace tfjs

#endif  // BINARY_H_
