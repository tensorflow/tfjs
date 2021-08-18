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
#include "tfjs-backend-wasm/src/cc/kernels/ClipByValue.h"

TEST(ClipByValue, xnn_operator_count) {
  tfjs::wasm::init();

  ASSERT_EQ(0, tfjs::backend::num_tensors());

  size_t x0_id = 1;
  size_t x1_id = 2;
  size_t size = 2;
  size_t min = 0;
  size_t max = 1;
  float x_values[2] = {1, 2};
  size_t out_id = 3;
  float out_values[2] = {0, 0};

  tfjs::wasm::register_tensor(x0_id, size, x_values);
  tfjs::wasm::register_tensor(x1_id, size, x_values);
  tfjs::wasm::register_tensor(out_id, size, out_values);

  ASSERT_EQ(3, tfjs::backend::num_tensors());
  ASSERT_EQ(0, tfjs::backend::xnn_operator_count);

  // One new xnn_operator should be created for the first call to clip.
  tfjs::wasm::ClipByValue(x0_id, min, max, out_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // No new xnn_operators should be created for the second call to clip with
  // the same min/max.
  tfjs::wasm::ClipByValue(x1_id, min, max, out_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // One new xnn_operator should be created for another call to clip with new
  // min/max.
  tfjs::wasm::ClipByValue(x0_id, min, max + 1, out_id);
  ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  // No new xnn_operators should be created for the next call to prelu with
  // the same min/max.
  tfjs::wasm::ClipByValue(x1_id, min, max + 1, out_id);
  ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  // Disposing x's should not remove xnn operators.
  tfjs::wasm::dispose_data(x0_id);
  tfjs::wasm::dispose_data(x1_id);
  ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  tfjs::wasm::dispose();
}
