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

#include <string.h>

#include "src/cc/backend.h"
#include "src/cc/kernels/Prelu.h"
#include "src/cc/util.h"

TEST(BACKEND, register_tensor) {
  tfjs::wasm::init(0);

  ASSERT_EQ(0, tfjs::backend::num_tensors());

  const int tensor_id = 0;
  const int size = 2;
  float values[size] = {1, 2};

  tfjs::wasm::register_tensor(tensor_id, size, values);
  ASSERT_EQ(1, tfjs::backend::num_tensors());

  auto& tensor_info = tfjs::backend::get_tensor_info(tensor_id);

  ASSERT_EQ(size, tensor_info.size);

  ASSERT_EQ(values, tensor_info.memory_offset);

  tfjs::wasm::dispose_data(tensor_id);

  ASSERT_EQ(0, tfjs::backend::num_tensors());

  tfjs::wasm::dispose();
}

// C++ doesn't allow lambda functions with captures so we define the callback
// outside the function. In the future we can consider changing the signature of
// register_disposal_callback to take a std::function.
int tensor_0_callback_count = 0;
int tensor_1_callback_count = 0;
void fake_dispose_tensor_callback(int tensor_id) {
  if (tensor_id == 0) {
    tensor_0_callback_count++;
  } else if (tensor_id == 1) {
    tensor_1_callback_count++;
  }
}
TEST(BACKEND, disposal_callback) {
  tfjs::wasm::init(0);

  ASSERT_EQ(0, tfjs::backend::num_tensors());

  const int tensor_id_0 = 0;
  const int tensor_id_1 = 1;
  const int size = 2;
  float values_0[size] = {1, 2};
  float values_1[size] = {3, 4};

  tfjs::wasm::register_tensor(tensor_id_0, size, values_0);
  tfjs::wasm::register_tensor(tensor_id_1, size, values_1);

  // Register two disposal callbacks on 0 but not 1.
  tfjs::backend::register_disposal_callback(tensor_id_0,
                                            *fake_dispose_tensor_callback);
  tfjs::backend::register_disposal_callback(tensor_id_0,
                                            *fake_dispose_tensor_callback);

  tfjs::wasm::dispose_data(tensor_id_0);

  ASSERT_EQ(2, tensor_0_callback_count);
  ASSERT_EQ(0, tensor_1_callback_count);

  tfjs::wasm::dispose_data(tensor_id_1);

  ASSERT_EQ(2, tensor_0_callback_count);
  ASSERT_EQ(0, tensor_1_callback_count);

  ASSERT_EQ(0, tfjs::backend::num_tensors());

  tfjs::wasm::dispose();

  tensor_0_callback_count = 0;
  tensor_1_callback_count = 0;
}

// 100 is longer than the longest string for error messages used.
char last_str[200];
int last_str_size = -1;
void throw_js_exception_fake(char* str, int str_size) {
  strcpy(last_str, str);
  last_str_size = str_size;
}

TEST(BACKEND, throw_js_exception) {
  tfjs::backend::set_throw_js_exception_fn(&throw_js_exception_fake);

  ASSERT_STREQ(last_str, "");
  ASSERT_EQ(last_str_size, -1);

  tfjs::backend::throw_js_exception("fake error message %d", 22);
  ASSERT_STREQ("fake error message 22", last_str);

  tfjs::wasm::dispose();
}
