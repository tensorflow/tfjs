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

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include "src/cc/kernels/Prelu.h"

#include <xnnpack.h>
#include <cmath>
#include <limits>
#include <unordered_map>

#include "src/cc/backend.h"
#include "src/cc/util.h"

namespace {
// The operator cache maps the weights id to the xnn_operator_t instantiated for
// // this set of weights.
std::unordered_map<int, xnn_operator_t> operator_cache;

void delete_xnn_operator(int weights_id) {
  xnn_operator_t prelu_op = operator_cache.at(weights_id);
  xnn_delete_operator(prelu_op);
  tfjs::backend::xnn_operator_count--;

  operator_cache.erase(weights_id);
}
}  // namespace

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void Prelu(int x_id, int x_size, int weights_id, int out_id) {
  const TensorInfo x_info = backend::get_tensor_info(x_id);
  const TensorInfo weights_info = backend::get_tensor_info(weights_id);
  const TensorInfo out_info = backend::get_tensor_info(out_id);

  const float* x_buf = static_cast<float*>(x_info.memory_offset);
  const float* weights_buf = static_cast<float*>(weights_info.memory_offset);
  float* out_buf = static_cast<float*>(out_info.memory_offset);

  xnn_operator_t prelu_op = nullptr;

  auto operator_cache_idx = operator_cache.find(weights_id);
  if (operator_cache_idx == operator_cache.end()) {
    int channels = x_size;
    int strides = channels;
    float output_min = -std::numeric_limits<float>::infinity();
    float output_max = std::numeric_limits<float>::infinity();

    const int flags = 0;
    xnn_status status =
        xnn_create_prelu_nc_f32(channels, strides, strides, weights_buf,
                                output_min, output_max, flags, &prelu_op);
    if (status != xnn_status_success) {
      util::warn(
          "XNN status for xnn_create_prelu_nc_f32 is not successful. Got "
          "status %d. Use -c dbg to see XNN logs.",
          status);
    }

    operator_cache.insert({weights_id, prelu_op});

    backend::register_disposal_callback(weights_id, *delete_xnn_operator);

    tfjs::backend::xnn_operator_count++;
  } else {
    prelu_op = operator_cache_idx->second;
  }

  const int batch_size = 1;
  xnn_status status = xnn_setup_prelu_nc_f32(
      prelu_op, batch_size, x_buf, out_buf, nullptr /* thread pool */);
  if (status != xnn_status_success) {
    util::warn(
        "XNN status for xnn_setup_prelu_nc_f32 is not successful. Got "
        "status %d. Use -c dbg to see XNN logs.",
        status);
  }

  xnn_run_operator(prelu_op, nullptr /* thread pool */);
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
