/* Copyright 2020 Google LLC. All Rights Reserved.
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

#include "tfjs-backend-wasm/src/cc/unary.h"

#include <xnnpack.h>
#include <cstddef>
#include <limits>
#include <unordered_map>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace {
// Maps an `xnn_create_*_nc_f32` function pointer to an instantiated operator.
std::unordered_map<tfjs::wasm::xnn_create_unary_op, xnn_operator_t> op_cache;
}  // namespace

namespace tfjs {
namespace wasm {

void unary_xnn_f32(const size_t x_id, const size_t out_id,
                   xnn_create_unary_op create_op, xnn_setup_unary_op setup_op) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  const float* x_buf = x_info.f32();
  float* out_buf = out_info.f32_write();

  xnn_operator_t unary_op = nullptr;

  auto cache_result = op_cache.find(create_op);
  if (cache_result == op_cache.end()) {
    const size_t channels = 1, input_stride = 1, output_stride = 1;
    const uint32_t flags = 1;
    xnn_status status =
        create_op(channels, input_stride, output_stride, flags, &unary_op);
    if (status != xnn_status_success) {
      util::warn(
          "XNN status for xnn_create_*_nd_f32 is not successful. Got "
          "status %d. Use -c dbg to see XNN logs.");
      return;
    }
    op_cache.insert({create_op, unary_op});
    backend::xnn_operator_count++;
  } else {
    unary_op = cache_result->second;
  }
  const size_t batch_size = out_info.size;
  xnn_status status =
      setup_op(unary_op, batch_size, x_buf, out_buf, tfjs::backend::threadpool);
  if (status != xnn_status_success) {
    util::warn(
        "XNN status for xnn_setup_*_nd_f32 is not successful. Got "
        "status %d. Use -c dbg to see XNN logs.",
        status);
    return;
  }

  xnn_run_operator(unary_op, tfjs::backend::threadpool);
}

}  // namespace wasm
}  // namespace tfjs
