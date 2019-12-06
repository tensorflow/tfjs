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
#include <xnnpack.h>

#include "src/cc/backend.h"
#include "src/cc/conv2d_impl.h"
#include "src/cc/kernels/FusedDepthwiseConv2D.h"
#include "src/cc/util.h"

TEST(FUSEDDEPTHWISECONV2D, xnn_operator_lifetime) {
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

  const int bias0_id = 4;
  const int bias1_id = 5;
  const int bias_size = 1;
  float bias_values[bias_size] = {1};

  const int out_id = 6;
  const int out_size = 12;
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
  const int batch_size = 1;
  const int input_height = 4;
  const int input_width = 2;
  const int filter_height = 4;
  const int filter_width = 2;
  const int pad_top0 = 1;
  const int pad_right = 0;
  const int pad_bottom0 = 0;
  const int pad_left = 0;
  const bool is_same_pad0 = false;
  const int dilation_height = 1;
  const int dilation_width = 1;
  const int stride_height = 1;
  const int stride_width = 1;
  const int input_channels = 1;
  const int output_channels = 1;

  const int activation = tfjs::wasm::FusableActivation::LINEAR;

  tfjs::wasm::FusedDepthwiseConv2D(
      x0_id, batch_size, input_height, input_width, weights0_id, filter_height,
      filter_width, -1 /* bias */, pad_top0, pad_right, pad_bottom0, pad_left,
      is_same_pad0, dilation_height, dilation_width, stride_height,
      stride_width, input_channels, output_channels, activation,
      -1 /* prelu weights */, out_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // One new xnn operator should be created for second call to conv2d with no
  // bias and prelu activation.
  const int prelu_activation = tfjs::wasm::FusableActivation::PRELU;

  const int prelu_weights_id = 7;
  const int prelu_size = 8;
  float prelu_values[prelu_size] = {1, 2, 3, 4, 5, 6, 7, 8};
  tfjs::wasm::register_tensor(prelu_weights_id, prelu_size, prelu_values);

  tfjs::wasm::FusedDepthwiseConv2D(
      x0_id, batch_size, input_height, input_width, weights0_id, filter_height,
      filter_width, -1 /* bias */, pad_top0, pad_right, pad_bottom0, pad_left,
      is_same_pad0, dilation_height, dilation_width, stride_height,
      stride_width, input_channels, output_channels, prelu_activation,
      prelu_weights_id, out_id);
  ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  // No new xnn_operators should be created for the second call to conv2d with
  // the same arguments.
  tfjs::wasm::FusedDepthwiseConv2D(
      x0_id, batch_size, input_height, input_width, weights0_id, filter_height,
      filter_width, -1 /* bias */, pad_top0, pad_right, pad_bottom0, pad_left,
      is_same_pad0, dilation_height, dilation_width, stride_height,
      stride_width, input_channels, output_channels, activation,
      prelu_weights_id, out_id);
  ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  // No new xnn_operators should be created for the second call to conv2d with
  // the same arguments but different input.
  tfjs::wasm::FusedDepthwiseConv2D(
      x1_id, batch_size, input_height, input_width, weights0_id, filter_height,
      filter_width, -1 /* bias */, pad_top0, pad_right, pad_bottom0, pad_left,
      is_same_pad0, dilation_height, dilation_width, stride_height,
      stride_width, input_channels, output_channels, activation,
      prelu_weights_id, out_id);
  ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  // One new xnn_operator should be created for the next call to conv2d with the
  // same weights and bias but different arguments.
  const int pad_top1 = 0;
  const int pad_bottom1 = 1;
  tfjs::wasm::FusedDepthwiseConv2D(
      x0_id, batch_size, input_height, input_width, weights0_id, filter_height,
      filter_width, -1 /* bias */, pad_top1, pad_right, pad_bottom1, pad_left,
      is_same_pad0, dilation_height, dilation_width, stride_height,
      stride_width, input_channels, output_channels, activation,
      prelu_weights_id, out_id);
  ASSERT_EQ(3, tfjs::backend::xnn_operator_count);

  // One more xnn operator should be created for the next call to conv2d with
  // new weights and same input.
  tfjs::wasm::FusedDepthwiseConv2D(
      x0_id, batch_size, input_height, input_width, weights1_id, filter_height,
      filter_width, -1 /* bias */, pad_top0, pad_right, pad_bottom0, pad_left,
      is_same_pad0, dilation_height, dilation_width, stride_height,
      stride_width, input_channels, output_channels, activation,
      prelu_weights_id, out_id);
  ASSERT_EQ(4, tfjs::backend::xnn_operator_count);

  // One more xnn operator should be created for the next call to conv2d with
  // bias, same input, same weights.
  tfjs::wasm::FusedDepthwiseConv2D(
      x0_id, batch_size, input_height, input_width, weights1_id, filter_height,
      filter_width, bias0_id, pad_top0, pad_right, pad_bottom0, pad_left,
      is_same_pad0, dilation_height, dilation_width, stride_height,
      stride_width, input_channels, output_channels, activation,
      prelu_weights_id, out_id);
  ASSERT_EQ(5, tfjs::backend::xnn_operator_count);

  // One more xnn operator should be created for the next call to conv2d with
  // new bias, same input, same weights.
  tfjs::wasm::FusedDepthwiseConv2D(
      x0_id, batch_size, input_height, input_width, weights1_id, filter_height,
      filter_width, bias1_id, pad_top0, pad_right, pad_bottom0, pad_left,
      is_same_pad0, dilation_height, dilation_width, stride_height,
      stride_width, input_channels, output_channels, activation,
      prelu_weights_id, out_id);
  ASSERT_EQ(6, tfjs::backend::xnn_operator_count);

  // One more xnn operator should be created for the next call to conv2d with
  // 'SAME' padding.
  const bool is_same_pad1 = true;
  tfjs::wasm::FusedDepthwiseConv2D(
      x0_id, batch_size, input_height, input_width, weights1_id, filter_height,
      filter_width, bias1_id, pad_top0, pad_right, pad_bottom0, pad_left,
      is_same_pad1, dilation_height, dilation_width, stride_height,
      stride_width, input_channels, output_channels, activation,
      prelu_weights_id, out_id);
  ASSERT_EQ(7, tfjs::backend::xnn_operator_count);

  // No new XNN operators should be created for the next call to conv2d with
  // 'SAME' padding and different input and raw padding values.
  tfjs::wasm::FusedDepthwiseConv2D(
      x1_id, batch_size, input_height, input_width, weights1_id, filter_height,
      filter_width, bias1_id, pad_top1, pad_right, pad_bottom1, pad_left,
      is_same_pad1, dilation_height, dilation_width, stride_height,
      stride_width, input_channels, output_channels, activation,
      prelu_weights_id, out_id);
  ASSERT_EQ(7, tfjs::backend::xnn_operator_count);

  // One new XNN operator should be created for the next call to conv2d with a
  // different activation.
  const int activation2 = tfjs::wasm::FusableActivation::RELU6;
  tfjs::wasm::FusedDepthwiseConv2D(
      x1_id, batch_size, input_height, input_width, weights1_id, filter_height,
      filter_width, bias1_id, pad_top1, pad_right, pad_bottom1, pad_left,
      is_same_pad1, dilation_height, dilation_width, stride_height,
      stride_width, input_channels, output_channels, activation2,
      prelu_weights_id, out_id);
  ASSERT_EQ(8, tfjs::backend::xnn_operator_count);

  // Disposing the first weights should remove 2 operators.
  tfjs::wasm::dispose_data(weights0_id);
  ASSERT_EQ(6, tfjs::backend::xnn_operator_count);

  // Disposing the second bias should remove 2 operators it's associated with.
  tfjs::wasm::dispose_data(bias1_id);
  ASSERT_EQ(3, tfjs::backend::xnn_operator_count);

  // Disposing the first bias should remove one operator.
  tfjs::wasm::dispose_data(bias0_id);
  ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  // Disposing the second weights should remove one operator.
  tfjs::wasm::dispose_data(weights1_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // Disposing prelu weights should remove the last operator.
  tfjs::wasm::dispose_data(prelu_weights_id);
  ASSERT_EQ(0, tfjs::backend::xnn_operator_count);

  tfjs::wasm::dispose();
}
