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
#include <vector>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/kernels/ResizeBilinear.h"

TEST(BATCH_MATMUL, xnn_operator_lifetime) {
  tfjs::wasm::init();

  ASSERT_EQ(0, tfjs::backend::num_tensors());

  const size_t x0_id = 1;
  const size_t x1_id = 2;
  const size_t x_size = 4;
  float x_values[x_size] = {1, 2, 2, 2};

  const size_t out_id = 3;
  const size_t out_size = 8;
  float out_values[out_size] = {0, 0, 0, 0, 0, 0, 0, 0};

  tfjs::wasm::register_tensor(x0_id, x_size, x_values);
  tfjs::wasm::register_tensor(x1_id, x_size, x_values);
  tfjs::wasm::register_tensor(out_id, out_size, out_values);

  ASSERT_EQ(3, tfjs::backend::num_tensors());
  ASSERT_EQ(0, tfjs::backend::xnn_operator_count);

  const size_t batch_size = 1;
  // One new xnn_operator should be created for the first call to
  // ResizeBilinear.
  const size_t old_height0 = 2;
  const size_t old_width0 = 1;
  const size_t num_channels0 = 2;
  const size_t new_height0 = 2;
  const size_t new_width0 = 2;
  bool align_corners0 = false;
  bool half_pixel_centers0 = false;
  tfjs::wasm::ResizeBilinear(x0_id, batch_size, old_height0, old_width0,
                             num_channels0, new_height0, new_width0,
                             align_corners0, half_pixel_centers0, out_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // No new xnn_operators should be created for the second call to
  // ResizeBilinear with the same arguments.
  tfjs::wasm::ResizeBilinear(x0_id, batch_size, old_height0, old_width0,
                             num_channels0, new_height0, new_width0,
                             align_corners0, half_pixel_centers0, out_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // No new xnn_operators should be created for the second call to
  // ResizeBilinear with a new x id but same arguments.
  tfjs::wasm::ResizeBilinear(x1_id, batch_size, old_height0, old_width0,
                             num_channels0, new_height0, new_width0,
                             align_corners0, half_pixel_centers0, out_id);
  ASSERT_EQ(1, tfjs::backend::xnn_operator_count);

  // One new xnn_operator should be created for another call to ResizeBilinear
  // with a different num_channels argument.
  const size_t old_height1 = 2;
  const size_t old_width1 = 2;
  const size_t num_channels1 = 1;
  const size_t new_height1 = 2;
  const size_t new_width1 = 4;
  tfjs::wasm::ResizeBilinear(x0_id, batch_size, old_height1, old_width1,
                             num_channels1, new_height1, new_width1,
                             align_corners0, half_pixel_centers0, out_id);
  ASSERT_EQ(2, tfjs::backend::xnn_operator_count);

  // One new xnn_operator should be created for another call to ResizeBilinear
  // with a different align_corners argument
  bool align_corners1 = true;
  tfjs::wasm::ResizeBilinear(x0_id, batch_size, old_height1, old_width1,
                             num_channels1, new_height1, new_width1,
                             align_corners1, half_pixel_centers0, out_id);
  ASSERT_EQ(3, tfjs::backend::xnn_operator_count);

  tfjs::wasm::dispose();
}
