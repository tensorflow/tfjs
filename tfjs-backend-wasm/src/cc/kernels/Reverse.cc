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

#include <cstddef>
#include <vector>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void Reverse(const size_t x_id, const size_t* axes_ptr,
             const size_t axes_length, const size_t* out_shape_ptr,
             const size_t out_shape_length, const size_t out_id) {
  auto out_shape =
      std::vector<size_t>(out_shape_ptr, out_shape_ptr + out_shape_length);
  auto axes = std::vector<size_t>(axes_ptr, axes_ptr + axes_length);

  auto& x_info = backend::get_tensor_info(x_id);
  const float* x_buf = x_info.f32();

  auto& out_info = backend::get_tensor_info_out(out_id);
  float* out_buf = out_info.f32_write();

  size_t x_size = x_info.size;

  const std::vector<size_t> out_strides =
      tfjs::util::compute_strides(out_shape);

  for (size_t i = 0; i < x_size; ++i) {
    std::vector<size_t> in_loc = tfjs::util::offset_to_loc(i, out_strides);

    for (size_t ax_i = 0; ax_i < axes_length; ++ax_i) {
      size_t ax = axes[ax_i];
      in_loc[ax] = out_shape[ax] - 1 - in_loc[ax];
    }

    const size_t x_position = tfjs::util::loc_to_offset(in_loc, out_strides);
    out_buf[i] = x_buf[x_position];
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
