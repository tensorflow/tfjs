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

#include "src/cc/backend.h"
#include "src/cc/kernels/_FusedMatMul.h"

TEST(_FUSED_MATMUL, xnn_operator_lfietime) {
  tfjs::wasm::init();

  ASSERT_EQ(0, tfjs::backend::num_tensors());

  size_t a0_id = 1;
  size_t a1_id = 2;
  size_t size = 2;
  float a_values[2] = {1, 2};
  std::vector<size_t> a_shape = {1, 2, 1};
  size_t* a_shape_ptr = a_shape.data();

  size_t b0_id = 3;
  size_t b1_id = 4;
  float b_values[2] = {1, 2};
  std::vector<size_t> b_shape = {1, 1, 2};
  size_t* b_shape_ptr = b_shape.data();

  size_t out_id = 5;
  float out_values[2] = {0, 0};

  tfjs::wasm::register_tensor(a0_id, size, a_values);
  tfjs::wasm::register_tensor(a1_id, size, a_values);
  tfjs::wasm::register_tensor(b0_id, size, b_values);
  tfjs::wasm::register_tensor(b1_id, size, b_values);
  tfjs::wasm::register_tensor(out_id, size, out_values);

  ASSERT_EQ(5, tfjs::backend::num_tensors());
  ASSERT_EQ(0, tfjs::backend::xnn_operator_count);

  const FusableActivation activation = FusableActivation::LINEAR;
  size_t bias_id = 0;
  size_t prelu_weights_id = 0;

  // One new xnn_operator should be created for the first call to
  // _FusedMatMul with no bias.
  tfjs::wasm::_FusedMatMul(a0_id, a_shape_ptr, a_shape.size(), b0_id,
                           b_shape_ptr, b_shape.size(), false /* transpose_a */,
                           false /* transpose_b */, activation, bias_id,
                           prelu_weights_id, out_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // No new xnn_operators should be created for the second call to
  // _FusedMatMul with the same arguments.
  tfjs::wasm::_FusedMatMul(a0_id, a_shape_ptr, a_shape.size(), b0_id,
                           b_shape_ptr, b_shape.size(), false /* transpose_a */,
                           false /* transpose_b */, activation, bias_id,
                           prelu_weights_id, out_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // No new xnn_operators should be created for calling _FusedMatMul
  // with a new a.
  tfjs::wasm::_FusedMatMul(a1_id, a_shape_ptr, a_shape.size(), b0_id,
                           b_shape_ptr, b_shape.size(), false /* transpose_a */,
                           false /* transpose_b */, activation, bias_id,
                           prelu_weights_id, out_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // One new xnn_operator should be created for calling _FusedMatMul
  // with a new b.
  tfjs::wasm::_FusedMatMul(a0_id, a_shape_ptr, a_shape.size(), b1_id,
                           b_shape_ptr, b_shape.size(), false /* transpose_a */,
                           false /* transpose_b */, activation, bias_id,
                           prelu_weights_id, out_id);
  ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  // No new xnn_operators should be created for the next call to
  // _FusedMatMul with the same b.
  tfjs::wasm::_FusedMatMul(a0_id, a_shape_ptr, a_shape.size(), b1_id,
                           b_shape_ptr, b_shape.size(), false /* transpose_a */,
                           false /* transpose_b */, activation, bias_id,
                           prelu_weights_id, out_id);
  ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  const size_t bias1_id = 6;
  const size_t bias_size = 1;
  float bias_values[bias_size] = {1};
  tfjs::wasm::register_tensor(bias1_id, bias_size, bias_values);
  // One new xnn_operator should be created for calling _FusedMatMul with a
  // new bias.
  tfjs::wasm::_FusedMatMul(a0_id, a_shape_ptr, a_shape.size(), b1_id,
                           b_shape_ptr, b_shape.size(), false /* transpose_a */,
                           false /* transpose_b */, activation, bias1_id,
                           prelu_weights_id, out_id);
  ASSERT_EQ(3, tfjs::backend::xnn_operator_count);

  // One new xnn_operator should be created for calling _FusedMatMul with a
  // different activation.
  const FusableActivation activation2 = FusableActivation::RELU;
  tfjs::wasm::_FusedMatMul(a0_id, a_shape_ptr, a_shape.size(), b1_id,
                           b_shape_ptr, b_shape.size(), false /* transpose_a */,
                           false /* transpose_b */, activation2, bias1_id,
                           prelu_weights_id, out_id);
  ASSERT_EQ(4, tfjs::backend::xnn_operator_count);

  // Disposing a's should not remove xnn operators.
  tfjs::wasm::dispose_data(a0_id);
  tfjs::wasm::dispose_data(a1_id);
  ASSERT_EQ(4, tfjs::backend::xnn_operator_count);

  // Disposing the second bias should remove the xnn_operators it's associated
  // with.
  tfjs::wasm::dispose_data(bias1_id);
  ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  // Disposing b's should remove xnn operators.
  tfjs::wasm::dispose_data(b0_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  tfjs::wasm::dispose_data(b1_id);
  ASSERT_EQ(0, tfjs::backend::xnn_operator_count);

  tfjs::wasm::dispose();
}
