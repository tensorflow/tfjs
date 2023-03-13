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

#include <random>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/shape.h"

namespace tfjs::wasm {

// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

// REQUIRES
// - Tensor `probabilities` is produced from normalized `logits`.
// - Tensor `probabilities` must have dtype float32.
// - Tensor `out` must have dtype int32.
// - Tensor `probabilities` must have shape [batch_size, num_events].
// - Tensor `out` must have shape [batch_size, num_samples].
void Multinomial(const int probabilities_id, const int batch_size,
                 const int num_events, const int num_samples, const float seed,
                 const int out_id) {
  const TensorInfo& prob_info = backend::get_tensor_info(probabilities_id);
  TensorInfo& out_info = backend::get_tensor_info_out(out_id);
  Shape<int, 2> probs_shape({batch_size, num_events});
  Shape<int, 2> out_shape({batch_size, num_samples});
  const float* probs_buf = prob_info.f32();
  int* out_buf = out_info.i32_write();

  std::mt19937 gen(*reinterpret_cast<const int32_t*>(&seed));
  for (int b = 0; b < batch_size; ++b) {
    const float* weights_begin = probs_buf + probs_shape.offset({b, 0});
    const float* weights_end = weights_begin + num_events;
    std::discrete_distribution<int32_t> distribution(weights_begin,
                                                     weights_end);
    for (int i = 0; i < num_samples; ++i) {
      out_buf[out_shape.offset({b, i})] = distribution(gen);
    }
  }
}

}  // extern "C"
}  // namespace tfjs::wasm
