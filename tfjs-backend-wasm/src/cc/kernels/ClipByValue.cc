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

#include "tfjs-backend-wasm/src/cc/kernels/ClipByValue.h"

#include <xnnpack.h>
#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <unordered_map>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace {
// These float values are keys to creating the clip operator. We use
// std::array instead of a vanilla array as it implements the compare operator
// needed for std::map.
typedef std::array<float, 2> OperatorCacheKey;

// The operator cache maps the cache key to the xnn_operator_t instantiated for
// this set of arguments to the xnn_operator.
std::map<OperatorCacheKey, xnn_operator_t> operator_cache;
}  // namespace

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void ClipByValue(const size_t x_id, const float min, const float max,
                 const size_t out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info_out(out_id);

  const float* x_buf = x_info.f32();
  float* out_buf = out_info.f32_write();

  xnn_operator_t clamp_op = nullptr;
  OperatorCacheKey cache_key = {min, max};
  auto operator_cache_idx = operator_cache.find(cache_key);
  if (operator_cache_idx == operator_cache.end()) {
    const size_t channels = 1;
    const size_t strides = channels;
    const uint32_t flags = 0;
    xnn_status status = xnn_create_clamp_nc_f32(channels, strides, strides, min,
                                                max, flags, &clamp_op);
    if (status != xnn_status_success) {
      util::warn(
          "XNN status for xnn_create_clamp_nc_f32 is not successful. Got "
          "status %d. Use -c dbg to see XNN logs.",
          status);
    }
    operator_cache.emplace(cache_key, clamp_op);

    tfjs::backend::xnn_operator_count++;
  } else {
    clamp_op = operator_cache_idx->second;
  }

  const size_t batch_size = x_info.size;
  xnn_status status = xnn_setup_clamp_nc_f32(
      clamp_op, batch_size, x_buf, out_buf, tfjs::backend::threadpool);
  if (status != xnn_status_success) {
    util::warn(
        "XNN status for xnn_setup_clamp_nc_f32 is not successful. Got "
        "status %d. Use -c dbg to see XNN logs.",
        status);
  }

  xnn_run_operator(clamp_op, tfjs::backend::threadpool);
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
