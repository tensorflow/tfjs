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
inline void LRNGradImpl(const T* x_buf, const T* y_buf, const T* dy_buf,
                        T* dx_buf, int size, int channels, int depth_radius,
                        float bias, float alpha, float beta) {
  std::fill(dx_buf, dx_buf + size, 0);
  for (int offset = 0; offset < size; ++offset) {
    int current_channel = offset % channels;
    int depth_begin =
        offset - current_channel + std::max(0, current_channel - depth_radius);
    int depth_end = offset - current_channel +
                    std::min(channels, current_channel + depth_radius + 1);

    T norm = 0;
    for (int i = depth_begin; i < depth_end; ++i) {
      norm += static_cast<T>(std::pow(x_buf[i], 2));
    }
    norm = alpha * norm + bias;

    for (int i = depth_begin; i < depth_end; ++i) {
      T dyi = -2 * alpha * beta * x_buf[i] * y_buf[offset] / norm;
      if (offset == i) {
        dyi += static_cast<T>(std::pow(norm, -beta));
      }
      dyi *= dy_buf[offset];
      dx_buf[i] += dyi;
    }
  }
}

}  // namespace

// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

// REQUIRES
// - Tensor `x`, `y`, 'dx', and 'dy' must have dtype float32.
// - Tensor `x`, `y`, 'dx', and 'dy' must have the same size.
void LRNGrad(const int x_id, const int y_id, const int dy_id, const int dx_id,
             const int channels, const int depth_radius, const float bias,
             const float alpha, const float beta) {
  const TensorInfo& x_info = backend::get_tensor_info(x_id);
  const TensorInfo& y_info = backend::get_tensor_info(y_id);
  const TensorInfo& dy_info = backend::get_tensor_info(dy_id);
  TensorInfo& dx_info = backend::get_tensor_info_out(dx_id);

  LRNGradImpl<float>(x_info.f32(), y_info.f32(), dy_info.f32(),
                     dx_info.f32_write(), x_info.size, channels, depth_radius,
                     bias, alpha, beta);
}

}  // extern "C"
}  // namespace tfjs::wasm
