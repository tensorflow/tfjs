
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

#include "src/cc/backend.h"
#include "src/cc/kernels/AvgPool.h"

TEST(MAXPOOL, xnn_operator_lifetime) {
  tfjs::wasm::init();

  ASSERT_EQ(0, tfjs::backend::num_tensors());

  const size_t x0_id = 1;
  const size_t x1_id = 2;
  const size_t size = 9;
  float x_values[size] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  const size_t out_id = 3;
  const size_t out_size = 9;
  float out_values[out_size] = {};

  tfjs::wasm::register_tensor(x0_id, size, x_values);
  tfjs::wasm::register_tensor(x1_id, size, x_values);
  tfjs::wasm::register_tensor(out_id, out_size, out_values);

  ASSERT_EQ(3, tfjs::backend::num_tensors());
  ASSERT_EQ(0, tfjs::backend::xnn_operator_count);

  // One xnn_operator should be created for first call to avgPool.
  const size_t batch_size = 1;
  const size_t input_height = 3;
  const size_t input_width = 3;
  const size_t filter_height = 2;
  const size_t filter_width = 2;
  const size_t pad_top = 0;
  const size_t pad_right = 1;
  const size_t pad_bottom = 1;
  const size_t pad_left = 0;
  const size_t stride_height = 1;
  const size_t stride_width = 1;
  const size_t channels = 1;
  tfjs::wasm::AvgPool(x0_id, batch_size, input_height, input_width,
                      filter_height, filter_width, pad_top, pad_right,
                      pad_bottom, pad_left, stride_height, stride_width,
                      channels, out_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // No new xnn_operators should be created for the second call to avgPool with
  // the same arguments.
  tfjs::wasm::AvgPool(x0_id, batch_size, input_height, input_width,
                      filter_height, filter_width, pad_top, pad_right,
                      pad_bottom, pad_left, stride_height, stride_width,
                      channels, out_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // One new xnn_operator should be created for the next call to avgPool with
  // 'valid' padding.
  tfjs::wasm::AvgPool(x0_id, batch_size, input_height, input_width,
                      filter_height, filter_width, pad_top, 0 /* pad_right */,
                      0 /* pad_bottom */, pad_left, stride_height, stride_width,
                      channels, out_id);
  ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  tfjs::wasm::dispose();
}
