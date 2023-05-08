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

#ifndef BINARY_H_
#define BINARY_H_

#include <xnnpack.h>
#include <algorithm>
#include <cstddef>

#include "tfjs-backend-wasm/src/cc/backend.h"

namespace tfjs {
namespace wasm {

template <class I, class O>
void binary_impl(const I* a_buf, const size_t a_size, const I* b_buf,
                 const size_t b_size, O* out_buf, O operation(I, I),
                 const size_t* a_shape_ptr, const size_t a_rank,
                 const size_t* b_shape_ptr, const size_t b_rank);

void binary_f32(const int a_id, const size_t* a_shape_ptr,
                const size_t a_shape_len, const int b_id,
                const size_t* b_shape_ptr, const size_t b_shape_len,
                const size_t out_id, float operation(float, float));

void binary_i32(const int a_id, const size_t* a_shape_ptr,
                const size_t a_shape_len, const int b_id,
                const size_t* b_shape_ptr, const size_t b_shape_len,
                const size_t out_id, int operation(int, int));

void binary_bool(const int a_id, const size_t* a_shape_ptr,
                 const size_t a_shape_len, const int b_id,
                 const size_t* b_shape_ptr, const size_t b_shape_len,
                 const size_t out_id, bool operation(bool, bool));

void compare_f32(const int a_id, const size_t* a_shape_ptr,
                 const size_t a_shape_len, const int b_id,
                 const size_t* b_shape_ptr, const size_t b_shape_len,
                 const int out_id, bool operation(float, float));

void compare_i32(const int a_id, const size_t* a_shape_ptr,
                 const size_t a_shape_len, const int b_id,
                 const size_t* b_shape_ptr, const size_t b_shape_len,
                 const int out_id, bool operation(int, int));

void compare_bool(const int a_id, const size_t* a_shape_ptr,
                  const size_t a_shape_len, const int b_id,
                  const size_t* b_shape_ptr, const size_t b_shape_len,
                  const int out_id, bool operation(bool, bool));

void logical(const int a_id, const size_t* a_shape_ptr,
             const size_t a_shape_len, const int b_id,
             const size_t* b_shape_ptr, const size_t b_shape_len,
             const int out_id, bool operation(bool, bool));

typedef xnn_status (*xnn_create_binary_op)(float, float, uint32_t,
                                           xnn_operator_t*);
typedef xnn_status (*xnn_setup_binary_op)(xnn_operator_t, size_t, const size_t*,
                                          size_t, const size_t*, const float*,
                                          const float*, float*, pthreadpool_t);

void binary_xnn_f32(const size_t a_id, const size_t* a_shape_ptr,
                    const size_t a_shape_len, const size_t b_id,
                    const size_t* b_shape_ptr, const size_t b_shape_len,
                    const size_t out_id, xnn_create_binary_op create_op,
                    xnn_setup_binary_op setup_op);

}  // namespace wasm
}  // namespace tfjs

#endif  // BINARY_H_
