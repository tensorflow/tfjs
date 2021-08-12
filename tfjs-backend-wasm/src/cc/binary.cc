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

#include "tfjs-backend-wasm/src/cc/binary.h"

#include <xnnpack.h>
#include <cstddef>
#include <limits>
#include <unordered_map>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace {
// Maps an `xnn_create_*_nd_f32` function pointer to an instantiated operator.
std::unordered_map<tfjs::wasm::xnn_create_binary_op, xnn_operator_t> op_cache;
}  // namespace

namespace tfjs {
namespace wasm {

void binary_xnn_f32(const size_t a_id, const size_t* a_shape_ptr,
                    const size_t a_shape_len, const size_t b_id,
                    const size_t* b_shape_ptr, const size_t b_shape_len,
                    const size_t out_id, xnn_create_binary_op create_op,
                    xnn_setup_binary_op setup_op) {
  auto& a_info = backend::get_tensor_info(a_id);
  auto& b_info = backend::get_tensor_info(b_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  const float* a_buf = a_info.f32();
  const float* b_buf = b_info.f32();
  float* out_buf = out_info.f32_write();

  xnn_operator_t binary_op = nullptr;

  auto cache_result = op_cache.find(create_op);
  if (cache_result == op_cache.end()) {
    const float sum_min = -std::numeric_limits<float>::infinity(),
                sum_max = std::numeric_limits<float>::infinity();
    const uint32_t flags = 0;
    xnn_status status = create_op(sum_min, sum_max, flags, &binary_op);
    if (status != xnn_status_success) {
      util::warn(
          "XNN status for xnn_create_*_nd_f32 is not successful. Got "
          "status %d. Use -c dbg to see XNN logs.");
      return;
    }
    op_cache.insert({create_op, binary_op});
    backend::xnn_operator_count++;
  } else {
    binary_op = cache_result->second;
  }
  xnn_status status =
      setup_op(binary_op, a_shape_len, a_shape_ptr, b_shape_len, b_shape_ptr,
               a_buf, b_buf, out_buf, tfjs::backend::threadpool);
  if (status != xnn_status_success) {
    util::warn(
        "XNN status for xnn_setup_*_nd_f32 is not successful. Got "
        "status %d. Use -c dbg to see XNN logs.",
        status);
    return;
  }

  xnn_run_operator(binary_op, tfjs::backend::threadpool);
}

}  // namespace wasm
}  // namespace tfjs
