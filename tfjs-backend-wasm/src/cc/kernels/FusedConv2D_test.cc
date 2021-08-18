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
#include <xnnpack.h>

#include <cstddef>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/kernels/FusedConv2D.h"
#include "tfjs-backend-wasm/src/cc/util.h"

TEST(FUSEDCONV2D, xnn_operator_lifetime) {
  tfjs::wasm::init();

  ASSERT_EQ(0, tfjs::backend::num_tensors());

  const size_t x0_id = 1;
  const size_t x1_id = 2;
  const size_t size = 8;
  float x_values[size] = {1, 2, 3, 4, 5, 6, 7, 8};

  const size_t weights0_id = 3;
  const size_t weights1_id = 4;
  const size_t weights_size = 8;
  float weights_values[weights_size] = {1, 2, 3, 4, 5, 6, 7, 8};

  const size_t bias0_id = 5;
  const size_t bias1_id = 6;
  const size_t bias_size = 1;
  float bias_values[bias_size] = {1};

  const size_t out_id = 7;
  const size_t out_size = 12;
  float out_values[out_size] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  tfjs::wasm::register_tensor(x0_id, size, x_values);
  tfjs::wasm::register_tensor(x1_id, size, x_values);
  tfjs::wasm::register_tensor(weights0_id, weights_size, weights_values);
  tfjs::wasm::register_tensor(weights1_id, weights_size, weights_values);
  tfjs::wasm::register_tensor(bias0_id, bias_size, bias_values);
  tfjs::wasm::register_tensor(bias1_id, bias_size, bias_values);
  tfjs::wasm::register_tensor(out_id, out_size, out_values);

  ASSERT_EQ(7, tfjs::backend::num_tensors());
  ASSERT_EQ(0, tfjs::backend::xnn_operator_count);

  // One xnn_operator should be created for the first call to conv2d with no
  // bias.
  const size_t batch_size = 1;
  const size_t input_height = 4;
  const size_t input_width = 2;
  const size_t filter_height = 4;
  const size_t filter_width = 2;
  const size_t pad_top0 = 1;
  const size_t pad_right = 0;
  const size_t pad_bottom0 = 0;
  const size_t pad_left = 0;
  const bool is_same_pad0 = false;
  const size_t dilation_height = 1;
  const size_t dilation_width = 1;
  const size_t stride_height = 1;
  const size_t stride_width = 1;
  const size_t input_channels = 1;
  const size_t output_channels = 1;

  const FusableActivation activation = FusableActivation::LINEAR;
  const size_t prelu_weights_id = 0;

  tfjs::wasm::FusedConv2D(
      x0_id, batch_size, input_height, input_width, weights0_id, filter_height,
      filter_width, 0 /* bias */, pad_top0, pad_right, pad_bottom0, pad_left,
      is_same_pad0, dilation_height, dilation_width, stride_height,
      stride_width, input_channels, output_channels, activation,
      prelu_weights_id, 0 /* leakyrelu alpha */, out_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // No new xnn_operators should be created for the second call to conv2d with
  // the same arguments.
  tfjs::wasm::FusedConv2D(
      x0_id, batch_size, input_height, input_width, weights0_id, filter_height,
      filter_width, 0 /* bias */, pad_top0, pad_right, pad_bottom0, pad_left,
      is_same_pad0, dilation_height, dilation_width, stride_height,
      stride_width, input_channels, output_channels, activation,
      prelu_weights_id, 0 /* leakyrelu alpha */, out_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // No new xnn_operators should be created for the second call to conv2d with
  // the same arguments but different input.
  tfjs::wasm::FusedConv2D(
      x1_id, batch_size, input_height, input_width, weights0_id, filter_height,
      filter_width, 0 /* bias */, pad_top0, pad_right, pad_bottom0, pad_left,
      is_same_pad0, dilation_height, dilation_width, stride_height,
      stride_width, input_channels, output_channels, activation,
      prelu_weights_id, 0 /* leakyrelu alpha */, out_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // One new xnn_operator should be created for the next call to conv2d with the
  // same weights and bias but different arguments.
  const size_t pad_top1 = 0;
  const size_t pad_bottom1 = 1;
  tfjs::wasm::FusedConv2D(
      x0_id, batch_size, input_height, input_width, weights0_id, filter_height,
      filter_width, 0 /* bias */, pad_top1, pad_right, pad_bottom1, pad_left,
      is_same_pad0, dilation_height, dilation_width, stride_height,
      stride_width, input_channels, output_channels, activation,
      prelu_weights_id, 0 /* leakyrelu alpha */, out_id);
  ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  // One more xnn operator should be created for the next call to conv2d with
  // new weights and same input.
  tfjs::wasm::FusedConv2D(
      x0_id, batch_size, input_height, input_width, weights1_id, filter_height,
      filter_width, 0 /* bias */, pad_top0, pad_right, pad_bottom0, pad_left,
      is_same_pad0, dilation_height, dilation_width, stride_height,
      stride_width, input_channels, output_channels, activation,
      prelu_weights_id, 0 /* leakyrelu alpha */, out_id);
  ASSERT_EQ(3, tfjs::backend::xnn_operator_count);

  // One more xnn operator should be created for the next call to conv2d with
  // bias, same input, same weights.
  tfjs::wasm::FusedConv2D(
      x0_id, batch_size, input_height, input_width, weights1_id, filter_height,
      filter_width, bias0_id, pad_top0, pad_right, pad_bottom0, pad_left,
      is_same_pad0, dilation_height, dilation_width, stride_height,
      stride_width, input_channels, output_channels, activation,
      prelu_weights_id, 0 /* leakyrelu alpha */, out_id);
  ASSERT_EQ(4, tfjs::backend::xnn_operator_count);

  // One more xnn operator should be created for the next call to conv2d with
  // new bias, same input, same weights.
  tfjs::wasm::FusedConv2D(
      x0_id, batch_size, input_height, input_width, weights1_id, filter_height,
      filter_width, bias1_id, pad_top0, pad_right, pad_bottom0, pad_left,
      is_same_pad0, dilation_height, dilation_width, stride_height,
      stride_width, input_channels, output_channels, activation,
      prelu_weights_id, 0 /* leakyrelu alpha */, out_id);
  ASSERT_EQ(5, tfjs::backend::xnn_operator_count);

  // One more xnn operator should be created for the next call to conv2d with
  // 'SAME' padding.
  const bool is_same_pad1 = true;
  tfjs::wasm::FusedConv2D(
      x0_id, batch_size, input_height, input_width, weights1_id, filter_height,
      filter_width, bias1_id, pad_top0, pad_right, pad_bottom0, pad_left,
      is_same_pad1, dilation_height, dilation_width, stride_height,
      stride_width, input_channels, output_channels, activation,
      prelu_weights_id, 0 /* leakyrelu alpha */, out_id);
  ASSERT_EQ(6, tfjs::backend::xnn_operator_count);

  // No new XNN operators should be created for the next call to conv2d with
  // 'SAME' padding and different input and raw padding values.
  tfjs::wasm::FusedConv2D(
      x1_id, batch_size, input_height, input_width, weights1_id, filter_height,
      filter_width, bias1_id, pad_top1, pad_right, pad_bottom1, pad_left,
      is_same_pad1, dilation_height, dilation_width, stride_height,
      stride_width, input_channels, output_channels, activation,
      prelu_weights_id, 0 /* leakyrelu alpha */, out_id);
  ASSERT_EQ(6, tfjs::backend::xnn_operator_count);

  // One new XNN operator should be created for the next call to conv2d with a
  // different activation.
  const FusableActivation activation2 = FusableActivation::RELU6;
  tfjs::wasm::FusedConv2D(
      x1_id, batch_size, input_height, input_width, weights1_id, filter_height,
      filter_width, bias1_id, pad_top1, pad_right, pad_bottom1, pad_left,
      is_same_pad1, dilation_height, dilation_width, stride_height,
      stride_width, input_channels, output_channels, activation2,
      prelu_weights_id, 0 /* leakyrelu alpha */, out_id);
  ASSERT_EQ(7, tfjs::backend::xnn_operator_count);

  // Disposing the first weights should remove 2 operators.
  tfjs::wasm::dispose_data(weights0_id);
  ASSERT_EQ(5, tfjs::backend::xnn_operator_count);

  // Disposing the second bias should remove 2 operators it's associated with.
  tfjs::wasm::dispose_data(bias1_id);
  ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  // Disposing the first bias should remove one operator.
  tfjs::wasm::dispose_data(bias0_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // Disposing the second weights should remove the last operator (no bias).
  tfjs::wasm::dispose_data(weights1_id);
  ASSERT_EQ(0, tfjs::backend::xnn_operator_count);

  tfjs::wasm::dispose();
}
