/* Copyright 2023 Google LLC.
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

#include <algorithm>
#include <cmath>

#include "tfjs-backend-wasm/src/cc/backend.h"

namespace tfjs::wasm {

namespace {

template <typename T>
inline void LRNImpl(const T* x_buf, T* out_buf, int size, int channels,
                    int depth_radius, float bias, float alpha, float beta) {
  for (int offset = 0; offset < size; ++offset) {
    int current_channel = offset % channels;
    int begin_sum_offset =
        offset - current_channel + std::max(0, current_channel - depth_radius);
    int end_sum_offset = offset - current_channel +
                         std::min(current_channel + depth_radius, channels - 1);

    T sum = 0;
    for (int i = begin_sum_offset; i <= end_sum_offset; ++i) {
      sum += x_buf[i] * x_buf[i];
    }
    T val = x_buf[offset] * static_cast<T>(std::pow(bias + alpha * sum, -beta));
    out_buf[offset] = val;
  }
}

}  // namespace

// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

// REQUIRES
// - Tensor `x` and `out` must have dtype float32.
// - Tensor `x` and `out` must have the same size.
void LRN(const int x_id, const int out_id, const int channels,
         const int depth_radius, const float bias, const float alpha,
         const float beta) {
  const TensorInfo& x_info = backend::get_tensor_info(x_id);
  TensorInfo& out_info = backend::get_tensor_info_out(out_id);

  LRNImpl<float>(x_info.f32(), out_info.f32_write(), x_info.size, channels,
                 depth_radius, bias, alpha, beta);
}

}  // extern "C"
}  // namespace tfjs::wasm
