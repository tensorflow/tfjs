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

#include "src/cc/backend.h"
#include "src/cc/kernels/Conv2D.h"

TEST(CONV2D, xnn_operator_lifetime) {
  tfjs::wasm::init();

  ASSERT_EQ(0, tfjs::backend::num_tensors());

  const int x0_id = 0;
  const int x1_id = 1;
  const int size = 8;
  float x_values[size] = {1, 2, 3, 4, 5, 6, 7, 8};

  const int weights0_id = 2;
  const int weights1_id = 3;
  const int weights_size = 8;
  float weights_values[weights_size] = {1, 2, 3, 4, 5, 6, 7, 8};

  const int out_id = 4;
  const int out_size = 12;
  float out_values[out_size] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  tfjs::wasm::register_tensor(x0_id, size, x_values);
  tfjs::wasm::register_tensor(x1_id, size, x_values);
  tfjs::wasm::register_tensor(weights0_id, weights_size, weights_values);
  tfjs::wasm::register_tensor(weights1_id, weights_size, weights_values);
  tfjs::wasm::register_tensor(out_id, out_size, out_values);

  ASSERT_EQ(5, tfjs::backend::num_tensors());
  ASSERT_EQ(0, tfjs::backend::xnn_operator_count);

  // One xnn_operator should be created for the first call to conv2d.
  const int batch_size = 1;
  const int input_height = 4;
  const int input_width = 2;
  const int filter_height = 4;
  const int filter_width = 2;
  const int pad_top0 = 1;
  const int pad_right = 0;
  const int pad_bottom0 = 0;
  const int pad_left = 0;
  const int is_same_pad0 = 0;
  const int dilation_height = 1;
  const int dilation_width = 1;
  const int stride_height = 1;
  const int stride_width = 1;
  const int input_channels = 1;
  const int output_channels = 1;
  tfjs::wasm::Conv2D(x0_id, batch_size, input_height, input_width, weights0_id,
                     filter_height, filter_width, pad_top0, pad_right,
                     pad_bottom0, pad_left, is_same_pad0, dilation_height,
                     dilation_width, stride_height, stride_width,
                     input_channels, output_channels, out_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // No new xnn_operators should be created for the second call to conv2d with
  // the same arguments.
  tfjs::wasm::Conv2D(x0_id, batch_size, input_height, input_width, weights0_id,
                     filter_height, filter_width, pad_top0, pad_right,
                     pad_bottom0, pad_left, is_same_pad0, dilation_height,
                     dilation_width, stride_height, stride_width,
                     input_channels, output_channels, out_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // No new xnn_operators should be created for the second call to conv2d with
  // the same arguments but different input.
  tfjs::wasm::Conv2D(x1_id, batch_size, input_height, input_width, weights0_id,
                     filter_height, filter_width, pad_top0, pad_right,
                     pad_bottom0, pad_left, is_same_pad0, dilation_height,
                     dilation_width, stride_height, stride_width,
                     input_channels, output_channels, out_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // One new xnn_operator should be created for the next call to conv2d with the
  // same weights but different arguments.
  const int pad_top1 = 0;
  const int pad_bottom1 = 1;
  tfjs::wasm::Conv2D(x0_id, batch_size, input_height, input_width, weights0_id,
                     filter_height, filter_width, pad_top1, pad_right,
                     pad_bottom1, pad_left, is_same_pad0, dilation_height,
                     dilation_width, stride_height, stride_width,
                     input_channels, output_channels, out_id);
  ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  // One more xnn operator should be created for the next call to conv2d with
  // new weights and same input.
  tfjs::wasm::Conv2D(x0_id, batch_size, input_height, input_width, weights1_id,
                     filter_height, filter_width, pad_top0, pad_right,
                     pad_bottom0, pad_left, is_same_pad0, dilation_height,
                     dilation_width, stride_height, stride_width,
                     input_channels, output_channels, out_id);
  ASSERT_EQ(3, tfjs::backend::xnn_operator_count);

  // One more xnn operator should be created for the next call to conv2d with
  // 'SAME' padding.
  const int is_same_pad1 = 1;
  tfjs::wasm::Conv2D(x0_id, batch_size, input_height, input_width, weights1_id,
                     filter_height, filter_width, pad_top0, pad_right,
                     pad_bottom0, pad_left, is_same_pad1, dilation_height,
                     dilation_width, stride_height, stride_width,
                     input_channels, output_channels, out_id);
  ASSERT_EQ(4, tfjs::backend::xnn_operator_count);

  // No new XNN operators should be created for the next call to conv2d with
  // 'SAME' padding and different input and raw padding values.
  tfjs::wasm::Conv2D(x1_id, batch_size, input_height, input_width, weights1_id,
                     filter_height, filter_width, pad_top1, pad_right,
                     pad_bottom1, pad_left, is_same_pad1, dilation_height,
                     dilation_width, stride_height, stride_width,
                     input_channels, output_channels, out_id);
  ASSERT_EQ(4, tfjs::backend::xnn_operator_count);

  // Disposing the first weights should remove 2 operators.
  tfjs::wasm::dispose_data(weights0_id);
  ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  // Disposing the second weights should remove the last 2 operator.
  tfjs::wasm::dispose_data(weights1_id);
  ASSERT_EQ(0, tfjs::backend::xnn_operator_count);

  tfjs::wasm::dispose();
}
