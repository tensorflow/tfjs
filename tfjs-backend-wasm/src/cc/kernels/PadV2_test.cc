/* Copyright 2020 Google LLC. All Rights Reserved.
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

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/kernels/PadV2.h"

TEST(PADV2, xnn_operator_lifetime) {
  tfjs::wasm::init();

  ASSERT_EQ(0, tfjs::backend::num_tensors());

  const size_t x0_id = 1;
  const size_t x1_id = 2;
  const size_t x_size = 4;
  float x_values[x_size] = {1, 2, 2, 2};

  const size_t out_id = 3;
  const size_t out_size = 8;
  float out_values[out_size] = {0, 0, 0, 0, 0, 0, 0, 0};

  tfjs::wasm::register_tensor(x0_id, x_size, x_values);
  tfjs::wasm::register_tensor(x1_id, x_size, x_values);
  tfjs::wasm::register_tensor(out_id, out_size, out_values);

  ASSERT_EQ(3, tfjs::backend::num_tensors());
  ASSERT_EQ(0, tfjs::backend::xnn_operator_count);

  const size_t x_rank = 4;

  std::vector<size_t> x_shape = {1, 1, 1, 4};
  std::vector<size_t> pre_paddings = {0, 0, 0, 0};
  std::vector<size_t> post_paddings = {0, 0, 0, 0};

  const float pad_value = 0.0;

  size_t* x_shape_ptr = x_shape.data();
  size_t* pre_paddings_ptr = pre_paddings.data();
  size_t* post_paddings_ptr = post_paddings.data();

  const DType dtype = float32;

  // One new xnn_operator should be created for the first call to
  // PadV2.
  tfjs::wasm::PadV2(x0_id, x_shape_ptr, x_rank, dtype, pre_paddings_ptr,
                    post_paddings_ptr, pad_value, out_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // // No new xnn_operators should be created for the second call to
  // // PadV2 with the same arguments.
  tfjs::wasm::PadV2(x0_id, x_shape_ptr, x_rank, dtype, pre_paddings_ptr,
                    post_paddings_ptr, pad_value, out_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // // No new xnn_operators should be created for the second call to
  // // PadV2 with a new x id but same arguments.
  tfjs::wasm::PadV2(x1_id, x_shape_ptr, x_rank, dtype, pre_paddings_ptr,
                    post_paddings_ptr, pad_value, out_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // // One new xnn_operator should be created for another call to PadV2
  // // with a different pad_value.
  const float new_pad_value = 0.5;
  tfjs::wasm::PadV2(x0_id, x_shape_ptr, x_rank, dtype, pre_paddings_ptr,
                    post_paddings_ptr, new_pad_value, out_id);
  ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  tfjs::wasm::dispose();
}
