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

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include <cmath>
#include <vector>

#include "src/cc/backend.h"
#include "src/cc/fft_impl.h"

namespace tfjs {
namespace wasm {

void fft(const size_t real_input_id, const size_t imag_input_id,
         const size_t outer_dim, const size_t inner_dim,
         const size_t is_real_component, const size_t out_id) {
  auto& real_input_info = backend::get_tensor_info(real_input_id);
  const float* real_input_buf = real_input_info.f32();
  auto& imag_input_info = backend::get_tensor_info(imag_input_id);
  const float* imag_input_buf = imag_input_info.f32();

  auto& out_info = backend::get_tensor_info_out(out_id);
  float* out_buf_ptr = out_info.f32_write();
  const size_t input_size = real_input_info.size;

  const float exponent_multiplier = -2.0 * M_PI;

  for (size_t row = 0; row < outer_dim; ++row) {
    for (size_t col = 0; col < inner_dim; ++col) {
      float index_ratio = float(col) / float(inner_dim);
      float exponent_multiplier_times_index_ratio =
          exponent_multiplier * index_ratio;

      float result = 0.0;

      for (size_t i = 0; i < inner_dim; ++i) {
        float x = exponent_multiplier_times_index_ratio * float(i);
        float exp_r = cos(x);
        float exp_i = sin(x);
        float real = real_input_buf[row * inner_dim + i];
        float imag = imag_input_buf[row * inner_dim + i];

        if (is_real_component > 0) {
          result += real * exp_r - imag * exp_i;
        } else {
          result += real * exp_i + imag * exp_r;
        }
      }

      *out_buf_ptr = result;
      out_buf_ptr++;
    }
  }
}
}  // namespace wasm
}  // namespace tfjs
