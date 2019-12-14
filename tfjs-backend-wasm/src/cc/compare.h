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

#ifndef COMPARE_H_
#define COMPARE_H_

#include <algorithm>

#include "src/cc/backend.h"
#include "src/cc/binary.h"

namespace tfjs {
namespace wasm {

inline void compare_f32(const int a_id, const int b_id, const int out_id,
                        bool operation(float, float)) {
  auto& a_info = backend::get_tensor_info(a_id);
  auto& b_info = backend::get_tensor_info(b_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  binary_impl<float, bool>(a_info.f32(), a_info.size, b_info.f32(), b_info.size,
                           out_info.b_write(), operation);
}

inline void compare_i32(const int a_id, const int b_id, const int out_id,
                        bool operation(int, int)) {
  auto& a_info = backend::get_tensor_info(a_id);
  auto& b_info = backend::get_tensor_info(b_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  binary_impl<int, bool>(a_info.i32(), a_info.size, b_info.i32(), b_info.size,
                         out_info.b_write(), operation);
}

inline void compare_bool(const int a_id, const int b_id, const int out_id,
                         bool operation(bool, bool)) {
  auto& a_info = backend::get_tensor_info(a_id);
  auto& b_info = backend::get_tensor_info(b_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  binary_impl<bool, bool>(a_info.b(), a_info.size, b_info.b(), b_info.size,
                          out_info.b_write(), operation);
}

}  // namespace wasm
}  // namespace tfjs

#endif  // COMPARE_H_
