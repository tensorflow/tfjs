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

#include <xnnpack.h>
#include <cmath>
#include <cstddef>
#include <map>
#include <tuple>
#include <vector>

#include "src/cc/kernels/ResizeBilinear.h"

#include "src/cc/backend.h"
#include "src/cc/util.h"

namespace {
// We use std::tuple as the cache key as it implements the compare operator
// needed for std::map.
typedef std::tuple<size_t, uint32_t> OperatorCacheKey;

// The operator cache maps the weights id to the xnn_operator_t instantiated for
// this set of weights.
std::map<OperatorCacheKey, xnn_operator_t> operator_cache;

}  // namespace

namespace tfjs {
namespace wasm {

extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void ResizeBilinear(size_t x_id, size_t batch, size_t old_height,
                    size_t old_width, size_t num_channels, size_t new_height,
                    size_t new_width, bool align_corners, size_t out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info_out(out_id);

  const float* x_buf = x_info.f32();
  float* out_buf = out_info.f32_write();

  xnn_operator_t resize_bilinear_op = nullptr;

  const uint32_t flags = XNN_FLAG_TENSORFLOW_LEGACY_MODE |
                         (align_corners ? XNN_FLAG_ALIGN_CORNERS : 0);

  OperatorCacheKey cache_key = {num_channels, flags};

  auto operator_cache_idx = operator_cache.find(cache_key);
  if (operator_cache_idx == operator_cache.end()) {
    const size_t channels = num_channels;
    const size_t input_pixel_stride = num_channels;
    const size_t output_pixel_stride = num_channels;

    xnn_status status = xnn_create_resize_bilinear2d_nhwc_f32(
        channels, input_pixel_stride, output_pixel_stride, flags,
        &resize_bilinear_op);
    if (status != xnn_status_success) {
      tfjs::util::warn(
          "XNN status for xnn_create_resize_bilinear2d_nhwc_f32 is not "
          "successful. Got status %d. Use -c dbg to see XNN logs.",
          status);
      return;
    }

    operator_cache.insert({cache_key, resize_bilinear_op});

    tfjs::backend::xnn_operator_count++;
  } else {
    resize_bilinear_op = operator_cache_idx->second;
  }

  xnn_status status = xnn_setup_resize_bilinear2d_nhwc_f32(
      resize_bilinear_op, batch, old_height, old_width, new_height, new_width,
      x_buf, out_buf, nullptr /* thread pool */);
  if (status != xnn_status_success) {
    tfjs::util::warn(
        "XNN status for xnn_setup_resize_bilinear2d_nhwc_f32 is not "
        "successful. Got status %d. Use -c dbg to see XNN logs.",
        status);
    return;
  }

  xnn_run_operator(resize_bilinear_op, nullptr /* thread pool */);
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
