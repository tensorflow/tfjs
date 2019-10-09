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

#include <emscripten.h>
#include <math.h>
#include <xnnpack.h>
#include <algorithm>
#include <cstdio>
#include <map>
#include <vector>

#include "src/cc/backend.h"
#include "src/cc/util.h"

const int kBlockSize = 48;

namespace tfjs {
// We use C-style API to interface with Javascript.
extern "C" {

EMSCRIPTEN_KEEPALIVE
void batch_matmul(int a_id, int b_id, int shared_dim, int left_dim,
                  int right_dim, int batch_dim, int a_batch, int a_outer_step,
                  int a_inner_step, int b_batch, int b_outer_step,
                  int b_inner_step, int out_id) {
  const TensorInfo a_info = backend::get_tensor_info(a_id);
  const TensorInfo b_info = backend::get_tensor_info(b_id);
  const TensorInfo out_info = backend::get_tensor_info(out_id);

  if (a_info.dtype != DType::float32) {
    util::warn("batch_matmul for tensor ids %d and %d failed. Unknown dtype %d",
               a_id, b_id, a_info.dtype);
  }

  const float* a_buf = a_info.buf.f32;
  const float* b_buf = b_info.buf.f32;
  float* out_buf = out_info.buf.f32;

  xnn_operator_t fully_connected_op = nullptr;
  int channels = x_size;
  int strides = channels;
  float output_min = -std::numeric_limits<float>::infinity();
  float output_max = std::numeric_limits<float>::infinity();

  xnn_status status = xnn_create_fully_connected_nc_f32(
      channels, strides, strides, weights_buf, output_min, output_max, 0,
      &fully_connected_op);
  if (status != xnn_status_success) {
    util::warn(
        "XNN status for xnn_create_prelu_nc_f32 is not successful. Got "
        "status %d. Use -c dbg to see XNN logs.",
        status);
  }

  operator_cache.insert({weights_id, fully_connected_op});

  // backend::register_disposal_callback(weights_id, *delete_xnn_operator);

  int batch_size = 1;
  xnn_status status =
      xnn_setup_prelu_nc_f32(fully_connected_op, batch_size, x_buf, out_buf,
                             nullptr /* thread pool */);
  if (status != xnn_status_success) {
    util::warn(
        "XNN status for xnn_setup_prelu_nc_f32 is not successful. Got "
        "status %d. Use -c dbg to see XNN logs.",
        status);
  }

  xnn_run_operator(fully_connected_op, nullptr /* thread pool */);
}

}  // extern "C"
}  // namespace tfjs
