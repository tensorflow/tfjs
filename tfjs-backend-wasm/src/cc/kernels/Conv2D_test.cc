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

TEST(PRELU, xnn_operator_lifetime) {
  tfjs::wasm::init();

  ASSERT_EQ(0, tfjs::backend::num_tensors());

  int x0_id = 0;
  int x1_id = 1;
  int shape[3] = {4, 2, 1};
  int shape_length = 3;
  int size = 8;
  DType dtype = DType::float32;
  float x_values[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  // const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [4, 2, inputDepth]);
  // const w =
  //     tf.tensor4d([3, 1, 5, 0, 2, 7, 8, 9], [4, 2, inputDepth, outputDepth]);

  int weights0_id = 2;
  int weights1_id = 3;
  float weights_values[2] = {1, 2, 3, 4, 5, 6, 7, 8};
  int weights_shape[4] = {4, 2, 1, 1};
  int weights_shape_length = 4;

  int out_id = 5;
  float out_values[2] = {0, 0, 0, 0, 0, 0, 0, 0};
  int out_shape[3] = {4, 2, 1};
  int out_shape_length = 3;

  tfjs::wasm::register_tensor(x0_id, shape, shape_length, dtype, x_values);
  tfjs::wasm::register_tensor(x1_id, shape, shape_length, dtype, x_values);
  tfjs::wasm::register_tensor(weights0_id, weights_shape, weights_shape_length,
                              dtype, weights_values);
  tfjs::wasm::register_tensor(weights1_id, weights_shape, weights_shape_length,
                              dtype, weights_values);
  tfjs::wasm::register_tensor(out_id, out_shape, out_shape_length, dtype,
                              out_values);

  ASSERT_EQ(5, tfjs::backend::num_tensors());
  ASSERT_EQ(0, tfjs::backend::xnn_operator_count);

  // One new xnn_operator should be created for the first call to conv2d.
  int batch_size = shape[0];
  int input_height = shape[1];
  int input_width = shape[2];
  int filter_height = weights_shape[0];
  int filter_width = weights_shape[1];
  int pad_top = 0;
  int pad_right = 0;
  int pad_bottom = 0;
  int pad_left = 0;
  int dilation_height = 1;
  int dilation_width;
  int stride_height = 1;
  int stride_width = 1;
  int input_channels = shape[2];
  int output_channels = out_shape[2];
  // void Conv2D(int x_id, int batch_size, int input_height, int input_width,
  //           int filter_id, int filter_height, int filter_width, int pad_top,
  //           int pad_right, int pad_bottom, int pad_left, int dilation_height,
  //           int dilation_width, int stride_height, int stride_width,
  //           int input_channels, int output_channels, int out_id) {
  tfjs::wasm::Conv2D(x0_id, batch_size, input_height, input_width, filter_id,
                     filter_height, filter_width, pad_top, pad_right,
                     pad_bottom, pad_left, dilation_height, dilation_width,
                     stride_height, stride_width, input_channels,
                     output_channels, out_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // // No new xnn_operators should be created for the second call to prelu with
  // // the same weights.
  // tfjs::wasm::Prelu(x1_id, size, weights0_id, out_id);
  // ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // // One new xnn_operator should be created for another call to prelu with
  // new
  // // weights.
  // tfjs::wasm::Prelu(x0_id, size, weights1_id, out_id);
  // ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  // // No new xnn_operators should be created for the next call to prelu with
  // // the same weights.
  // tfjs::wasm::Prelu(x1_id, size, weights1_id, out_id);
  // ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  // // Disposing x's should not remove xnn operators.
  // tfjs::wasm::dispose_data(x0_id);
  // tfjs::wasm::dispose_data(x1_id);
  // ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  // // Disposing weights should remove xnn operators.
  // tfjs::wasm::dispose_data(weights0_id);
  // ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // tfjs::wasm::dispose_data(weights1_id);
  // ASSERT_EQ(0, tfjs::backend::xnn_operator_count);

  tfjs::wasm::dispose();
}
