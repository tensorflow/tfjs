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

#include <cmath>
#include <cstddef>
#include <vector>

#include "tfjs-backend-wasm/src/cc/backend.h"

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void FusedBatchNorm(const size_t x_id, const size_t mean_id,
                    const size_t variance_id, const size_t offset_id,
                    const size_t scale_id, const float variance_epsilon,
                    const size_t out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& mean_info = backend::get_tensor_info(mean_id);
  auto& variance_info = backend::get_tensor_info(variance_id);
  auto& out_info = backend::get_tensor_info_out(out_id);

  const float* x_buf = x_info.f32();
  const size_t x_size = x_info.size;
  const float* mean_buf = mean_info.f32();
  const size_t mean_size = mean_info.size;
  const float* variance_buf = variance_info.f32();
  const size_t variance_size = variance_info.size;

  float* out_buf = out_info.f32_write();

  size_t offset_i = 0;
  size_t mean_i = 0;
  size_t scale_i = 0;
  size_t variance_i = 0;

  const float scale_buf_default[1] = {1};
  const float* scale_buf;
  size_t scale_size;
  if (scale_id == 0) {
    scale_buf = scale_buf_default;
    scale_size = 1;
  } else {
    auto& scale_info = backend::get_tensor_info(scale_id);
    scale_buf = scale_info.f32();
    scale_size = scale_info.size;
  }

  const float offset_buf_default[1] = {0};
  const float* offset_buf;
  size_t offset_size;
  if (offset_id == 0) {
    offset_buf = offset_buf_default;
    offset_size = 1;
  } else {
    auto& offset_info = backend::get_tensor_info(offset_id);
    offset_buf = offset_info.f32();
    offset_size = offset_info.size;
  }

  std::vector<double> normalization_factor(variance_size);
  for (size_t i = 0; i < variance_size; ++i) {
    normalization_factor[i] = std::sqrt(variance_buf[i] + variance_epsilon);
  }

  for (size_t i = 0; i < x_size; ++i) {
    out_buf[i] = offset_buf[offset_i] + (x_buf[i] - mean_buf[mean_i]) *
                                            scale_buf[scale_i] /
                                            normalization_factor[variance_i];

    ++offset_i;
    ++mean_i;
    ++scale_i;
    ++variance_i;

    if (offset_i >= offset_size) {
      offset_i = 0;
    }
    if (mean_i >= mean_size) {
      mean_i = 0;
    }
    if (scale_i >= scale_size) {
      scale_i = 0;
    }
    if (variance_i >= variance_size) {
      variance_i = 0;
    }
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
