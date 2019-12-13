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

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "src/cc/backend.h"
#include "src/cc/kernels/ResizeBilinear.h"

TEST(BATCH_MATMUL, xnn_operator_lifetime) {
  tfjs::wasm::init();

  ASSERT_EQ(0, tfjs::backend::num_tensors());

  size_t x0_id = 1;
  size_t x1_id = 2;
  size_t x_size = 4;
  float x_values[2] = {1, 2, 2, 2};

  size_t out_id = 3;
  size_t out_size = 8;
  float out_values[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  tfjs::wasm::register_tensor(x0_id, x_size, x_values);
  tfjs::wasm::register_tensor(x1_id, x_size, x_values);
  tfjs::wasm::register_tensor(out_id, out_size, out_values);

  ASSERT_EQ(3, tfjs::backend::num_tensors());
  ASSERT_EQ(0, tfjs::backend::xnn_operator_count);

  size_t batch_size = 1;
  // One new xnn_operator should be created for the first call to
  // ResizeBilinear.
  size_t old_height0 = 2;
  size_t old_width0 = 1;
  size_t num_channels0 = 2;
  size_t new_height0 = 2;
  size_t new_width0 = 2;
  bool align_corners0 = false;
  bool tfjs::wasm::ResizeBilinear(x0_id, batch_size, old_height0, old_width0,
                                  num_channels0, new_height0, new_width0,
                                  align_corners, out_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // // No new xnn_operators should be created for the second call to
  // BatchMatMul
  // // with the same b's.
  // tfjs::wasm::BatchMatMul(x0_id, a_shape_ptr, a_shape.size(), b0_id,
  //                         b_shape_ptr, b_shape.size(), false /* transpose_a
  //                         */, false /* transpose_b */, out_id);
  // ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // // One new xnn_operator should be created for another call to BatchMatMul
  // with
  // // new b's.
  // tfjs::wasm::BatchMatMul(x0_id, a_shape_ptr, a_shape.size(), b1_id,
  //                         b_shape_ptr, b_shape.size(), false /* transpose_a
  //                         */, false /* transpose_b */, out_id);
  // ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  // // No new xnn_operators should be created for the next call to BatchMatMul
  // // with the same b's.
  // tfjs::wasm::BatchMatMul(x0_id, a_shape_ptr, a_shape.size(), b1_id,
  //                         b_shape_ptr, b_shape.size(), false /* transpose_a
  //                         */, false /* transpose_b */, out_id);
  // ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  // // Disposing a's should not remove xnn operators.
  // tfjs::wasm::dispose_data(x0_id);
  // tfjs::wasm::dispose_data(x1_id);
  // ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  // // Disposing b's should remove xnn operators.
  // tfjs::wasm::dispose_data(b0_id);
  // ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // tfjs::wasm::dispose_data(b1_id);
  // ASSERT_EQ(0, tfjs::backend::xnn_operator_count);

  tfjs::wasm::dispose();
}
