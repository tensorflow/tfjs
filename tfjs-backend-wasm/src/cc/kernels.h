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

#ifndef TFJS_WASM_KERNELS_H_
#define TFJS_WASM_KERNELS_H_

namespace tfjs {
namespace kernels {

template <class T>
// Element-wise add of two tensors.
void add(T* a_buf, int a_size, T* b_buf, int b_size, T* out_buf);

// Batched matrix multiply.
void batchMatMul(float* a_buf, float* b_buf, int shared_dim, int left_dim,
                 int right_dim, int batch_dim, int a_batch, int a_outer_step,
                 int a_inner_step, int b_batch, int b_outer_step,
                 int b_inner_step, float* out_buf);
}  // namespace kernels
}  // namespace tfjs

#endif  // TFJS_WASM_KERNELS_H_
