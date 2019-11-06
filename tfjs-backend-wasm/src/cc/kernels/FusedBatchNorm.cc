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

#include <cmath>
#include <vector>
#include "src/cc/backend.h"

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void FusedBatchNorm(const int x_id, const int mean_id, const int variance_id,
                    const int offset_id, const int scale_id,
                    const float variance_epsilon, const int out_id) {
  auto& x_info = backend::get_tensor_info(x_id);
  auto& mean_info = backend::get_tensor_info(mean_id);
  auto& variance_info = backend::get_tensor_info(variance_id);
  auto& out_info = backend::get_tensor_info(out_id);

  const float* x_buf = reinterpret_cast<const float*>(x_info.memory_offset);
  const int x_size = x_info.size;
  const float* mean_buf =
      reinterpret_cast<const float*>(mean_info.memory_offset);
  const int mean_size = mean_info.size;
  const float* variance_buf =
      reinterpret_cast<const float*>(variance_info.memory_offset);
  const int variance_size = variance_info.size;

  float* out_buf = reinterpret_cast<float*>(out_info.memory_offset);

  int offset_i = 0;
  int mean_i = 0;
  int scale_i = 0;
  int variance_i = 0;

  float scale_buf_default[1] = {1};
  float* scale_buf;
  int scale_size;
  if (scale_id < 0) {
    scale_buf = scale_buf_default;
    scale_size = 1;
  } else {
    auto& scale_info = backend::get_tensor_info(scale_id);
    scale_buf = reinterpret_cast<float*>(scale_info.memory_offset);
    scale_size = scale_info.size;
  }

  float offset_buf_default[1] = {0};
  float* offset_buf;
  int offset_size;
  if (offset_id < 0) {
    offset_buf = offset_buf_default;
    offset_size = 1;
  } else {
    auto& offset_info = backend::get_tensor_info(offset_id);
    offset_buf = reinterpret_cast<float*>(offset_info.memory_offset);
    offset_size = offset_info.size;
  }

  std::vector<double> normalization_factor(variance_size);
  for (int i = 0; i < variance_size; ++i) {
    normalization_factor[i] = std::sqrt(variance_buf[i] + variance_epsilon);
  }

  for (int i = 0; i < x_size; ++i) {
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
