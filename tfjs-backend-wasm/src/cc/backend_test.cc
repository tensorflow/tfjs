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

TEST(BACKEND, register_tensor) {
  ASSERT_EQ(0, tfjs::backend::num_tensors());

  int tensor_id = 0;
  int shape[2] = {1, 2};
  int shape_length = 2;
  DType dtype = DType::float32;
  float values[2] = {1, 2};

  tfjs::wasm::register_tensor(tensor_id, shape, shape_length, dtype, values);
  ASSERT_EQ(1, tfjs::backend::num_tensors());

  TensorInfo tensor_info = tfjs::backend::get_tensor_info(tensor_id);

  ASSERT_EQ(dtype, tensor_info.dtype);

  ASSERT_EQ(shape[0], tensor_info.shape[0]);
  ASSERT_EQ(shape[1], tensor_info.shape[1]);

  ASSERT_EQ(values[0], tensor_info.buf.f32[0]);
  ASSERT_EQ(values[1], tensor_info.buf.f32[1]);

  tfjs::wasm::dispose_data(tensor_id);
  ASSERT_EQ(0, tfjs::backend::num_tensors());
}
