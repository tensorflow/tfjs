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

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include "src/cc/prelu_impl.h"

#include <xnnpack.h>
#include <cmath>
#include <cstddef>
#include <limits>
#include <unordered_map>

#include "src/cc/backend.h"
#include "src/cc/util.h"

namespace {
// The operator cache maps the weights id to the xnn_operator_t instantiated for
// this set of weights.
std::unordered_map<size_t, xnn_operator_t> operator_cache;

void delete_xnn_operator(const size_t weights_id) {
  xnn_operator_t prelu_op = operator_cache.at(weights_id);
  xnn_delete_operator(prelu_op);
  tfjs::backend::xnn_operator_count--;

  operator_cache.erase(weights_id);
}
}  // namespace

namespace tfjs {
namespace wasm {

void prelu(const float* x_buf, const size_t x_size, const size_t weights_id,
           const size_t out_id) {
  auto& weights_info = backend::get_tensor_info(weights_id);
  auto& out_info = backend::get_tensor_info_out(out_id);

  const float* weights_buf = weights_info.f32();
  float* out_buf = out_info.f32_write();

  xnn_operator_t prelu_op = nullptr;

  auto operator_cache_idx = operator_cache.find(weights_id);
  if (operator_cache_idx == operator_cache.end()) {
    const size_t channels = weights_info.size;
    const size_t strides = channels;

    const uint32_t flags = 0;
    xnn_status status = xnn_create_prelu_nc_f32(channels, strides, strides,
                                                weights_buf, flags, &prelu_op);
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

  const size_t batch_size = x_size / weights_info.size;
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

}  // namespace wasm
}  // namespace tfjs
