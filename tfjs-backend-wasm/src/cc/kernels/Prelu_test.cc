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

#include <gtest/gtest.h>

#include <cstddef>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/kernels/Prelu.h"

TEST(PRELU, xnn_operator_lifetime) {
  tfjs::wasm::init();

  ASSERT_EQ(0, tfjs::backend::num_tensors());

  size_t x0_id = 1;
  size_t x1_id = 2;
  size_t size = 2;
  float x_values[2] = {1, 2};

  size_t weights0_id = 3;
  size_t weights1_id = 4;
  float weights_values[2] = {1, 2};

  size_t out_id = 5;
  float out_values[2] = {0, 0};

  tfjs::wasm::register_tensor(x0_id, size, x_values);
  tfjs::wasm::register_tensor(x1_id, size, x_values);
  tfjs::wasm::register_tensor(weights0_id, size, weights_values);
  tfjs::wasm::register_tensor(weights1_id, size, weights_values);
  tfjs::wasm::register_tensor(out_id, size, out_values);

  ASSERT_EQ(5, tfjs::backend::num_tensors());
  ASSERT_EQ(0, tfjs::backend::xnn_operator_count);

  // One new xnn_operator should be created for the first call to prelu.
  tfjs::wasm::Prelu(x0_id, weights0_id, out_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // No new xnn_operators should be created for the second call to prelu with
  // the same weights.
  tfjs::wasm::Prelu(x1_id, weights0_id, out_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // One new xnn_operator should be created for another call to prelu with new
  // weights.
  tfjs::wasm::Prelu(x0_id, weights1_id, out_id);
  ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  // No new xnn_operators should be created for the next call to prelu with
  // the same weights.
  tfjs::wasm::Prelu(x1_id, weights1_id, out_id);
  ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  // Disposing x's should not remove xnn operators.
  tfjs::wasm::dispose_data(x0_id);
  tfjs::wasm::dispose_data(x1_id);
  ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  // Disposing weights should remove xnn operators.
  tfjs::wasm::dispose_data(weights0_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  tfjs::wasm::dispose_data(weights1_id);
  ASSERT_EQ(0, tfjs::backend::xnn_operator_count);

  tfjs::wasm::dispose();
}
