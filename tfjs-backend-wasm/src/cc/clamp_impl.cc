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

#include "tfjs-backend-wasm/src/cc/clamp_impl.h"

#include <xnnpack.h>
#include <cstddef>
#include <map>
#include <tuple>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace {
// These values are keys to creating the xnn clamp operator. We use
// std::tuple since it implements the compare operator needed for std::map.
typedef std::tuple<float, float> CacheKey;
// The operator cache maps the params of xnn_create_clamp_nc_f32 to an operator.
std::map<CacheKey, xnn_operator_t> op_cache;
}  // namespace

namespace tfjs {
namespace wasm {

void xnn_clamp(const size_t x_id, const size_t out_id, const float min,
               const float max) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  const float* x_buf = x_info.f32();
  float* out_buf = out_info.f32_write();

  xnn_operator_t op = nullptr;
  CacheKey cache_key = {min, max};
  const auto& cache_result = op_cache.find(cache_key);
  if (cache_result == op_cache.end()) {
    const size_t channels = 1, input_stride = 1, output_stride = 1, flags = 1;
    xnn_status status = xnn_create_clamp_nc_f32(
        channels, input_stride, output_stride, min, max, flags, &op);
    if (status != xnn_status_success) {
      util::warn(
          "XNN status for xnn_create_clamp_nc_f32 is not successful. "
          "Got status %d. Use -c dbg to see XNN logs.",
          status);
      return;
    }
    op_cache.emplace(cache_key, op);
    backend::xnn_operator_count++;
  } else {
    op = cache_result->second;
  }

  const size_t batch_size = out_info.size;
  xnn_status status = xnn_setup_clamp_nc_f32(op, batch_size, x_buf, out_buf,
                                             tfjs::backend::threadpool);
  if (status != xnn_status_success) {
    util::warn(
        "XNN status for xnn_setup_clamp_nc_f32 is not successful. "
        "Got status %d. Use -c dbg to see XNN logs.",
        status);
    return;
  }

  xnn_run_operator(op, tfjs::backend::threadpool);
}

}  // namespace wasm
}  // namespace tfjs
