/* Copyright 2019 Google LLC. All Rights Reserved.
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

#ifndef UNARY_H_
#define UNARY_H_

#include <xnnpack.h>
#include <cstddef>

#include "src/cc/backend.h"

namespace tfjs {
namespace wasm {

inline void unary(const size_t x_id, const size_t out_id,
                  float operation(float)) {
  auto& a_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info_out(out_id);

  const float* a_buf = a_info.f32();
  float* out_buf = out_info.f32_write();

  for (size_t i = 0; i < a_info.size; ++i) {
    out_buf[i] = operation(a_buf[i]);
  }
}

typedef xnn_status (*xnn_create_unary_op)(size_t, size_t, size_t, uint32_t,
                                          xnn_operator_t*);
typedef xnn_status (*xnn_setup_unary_op)(xnn_operator_t, size_t, const float*,
                                         float*, pthreadpool_t);

void unary_xnn_f32(const size_t x_id, const size_t out_id,
                   xnn_create_unary_op create_op, xnn_setup_unary_op setup_op);

}  // namespace wasm
}  // namespace tfjs

#endif  // UNARY_H_
