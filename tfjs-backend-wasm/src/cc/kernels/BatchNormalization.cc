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

#include <math.h>
#include <vector>

#include "src/cc/backend.h"
#include "src/cc/util.h"

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void BatchNormalization(int x_id, int mean_id, int variance_id, int offset_id,
                        int scale_id, float variance_epsilon, int out_id) {
  const auto x_info = backend::get_tensor_info(x_id);
  const auto mean_info = backend::get_tensor_info(mean_id);
  const auto variance_info = backend::get_tensor_info(variance_id);
  const auto out_info = backend::get_tensor_info(out_id);

  float* x_buf = x_info.buf.f32;
  int x_size = x_info.size;
  float* mean_buf = mean_info.buf.f32;
  int mean_size = mean_info.size;
  float* variance_buf = variance_info.buf.f32;
  int variance_size = variance_info.size;

  float* out_buf = out_info.buf.f32;

  int offi = 0;
  int mi = 0;
  int si = 0;
  int vi = 0;

  float scale_buf_default[1] = {1};
  float* scale_buf;
  int scale_size;
  if (scale_id < 0) {
    scale_buf = scale_buf_default;
    scale_size = 1;
  } else {
    const auto scale_info = backend::get_tensor_info(scale_id);
    scale_buf = scale_info.buf.f32;
    scale_size = scale_info.size;
  }

  float offset_buf_default[1] = {0};
  float* offset_buf;
  int offset_size;
  if (offset_id < 0) {
    offset_buf = offset_buf_default;
    offset_size = 1;
  } else {
    const auto offset_info = backend::get_tensor_info(offset_id);
    offset_buf = offset_info.buf.f32;
    offset_size = offset_info.size;
  }

  for (int i = 0; i < x_size; ++i) {
    out_buf[i] =
        offset_buf[offi] + (x_buf[i] - mean_buf[mi]) * scale_buf[si] /
                               sqrt(variance_buf[vi] + variance_epsilon);

    offi = offi + 1;
    mi = mi + 1;
    si = si + 1;
    vi = vi + 1;

    if (offi >= offset_size) {
      offi = 0;
    }
    if (mi >= mean_size) {
      mi = 0;
    }
    if (si >= scale_size) {
      si = 0;
    }
    if (vi >= variance_size) {
      vi = 0;
    }
  }
}

}  // namespace wasm
}  // namespace wasm
}  // namespace tfjs
