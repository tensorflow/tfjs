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
#include <xnnpack.h>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <vector>

#include "src/cc/backend.h"
#include "src/cc/util.h"

const int kBlockSize = 48;

namespace {
std::map<int, xnn_operator_t> operator_cache;
}  // namespace

namespace tfjs {
// We use C-style API to interface with Javascript.
extern "C" {

EMSCRIPTEN_KEEPALIVE
void prelu(int x_id, int x_size, int weights_id, int out_id) {
  const TensorInfo x_info = backend::get_tensor_info(x_id);
  const TensorInfo weights_info = backend::get_tensor_info(weights_id);
  const TensorInfo out_info = backend::get_tensor_info(out_id);

  const float* x_buf = x_info.buf.f32;
  const float* weights_buf = weights_info.buf.f32;
  float* out_buf = out_info.buf.f32;

  xnn_operator_t prelu_op = nullptr;
  if (operator_cache.count(weights_id) == 0) {
    int channels = x_size;
    int strides = channels;
    float output_min = -std::numeric_limits<float>::infinity();
    float output_max = std::numeric_limits<float>::infinity();

    xnn_status status =
        xnn_create_prelu_nc_f32(channels, strides, strides, weights_buf,
                                output_min, output_max, 0, &prelu_op);
    if (status != xnn_status_success) {
      util::warn("Bad XNN status for xnn_create_prelu_nc_f32, status is %d",
                 status);
    }

    operator_cache.insert({weights_id, prelu_op});
  } else {
    prelu_op = operator_cache.at(weights_id);
  }

  int batch_size = 1;
  xnn_status status = xnn_setup_prelu_nc_f32(
      prelu_op, batch_size, x_buf, out_buf, nullptr /* thread pool */);
  if (status != xnn_status_success) {
    util::warn("Bad XNN status for xnn_setup_prelu_nc_f32, status is %d",
               status);
  }

  xnn_run_operator(prelu_op, nullptr /* thread pool */);
}

}  // extern "C"
}  // namespace tfjs
