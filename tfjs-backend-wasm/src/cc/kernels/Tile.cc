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

namespace {
template <typename T>
void tile_slow(const T* x_data, const std::vector<size_t>& x_shape,
               const std::vector<size_t>& new_shape, T* out_data) {
  const size_t x_rank = x_shape.size();
  const std::vector<size_t> x_strides = tfjs::util::compute_strides(x_shape);

  const size_t out_size = tfjs::util::size_from_shape(new_shape);
  const std::vector<size_t> out_strides =
      tfjs::util::compute_strides(new_shape);

  for (size_t i = 0; i < out_size; ++i) {
    const std::vector<size_t> new_loc =
        tfjs::util::offset_to_loc(i, out_strides);

    std::vector<size_t> original_loc(x_rank);

    for (size_t j = 0; j < original_loc.size(); ++j) {
      original_loc[j] = new_loc[j] % x_shape[j];
    }

    const size_t original_index =
        tfjs::util::loc_to_offset(original_loc, x_strides);

    out_data[i] = x_data[original_index];
  }
}
}  // namespace

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void Tile(const size_t x_id, const size_t* x_shape_ptr,
          const size_t x_shape_length, const size_t* new_shape_ptr,
          const size_t new_shape_length, const DType dtype,
          const size_t out_id) {
  auto x_shape = std::vector<size_t>(x_shape_ptr, x_shape_ptr + x_shape_length);
  auto new_shape =
      std::vector<size_t>(new_shape_ptr, new_shape_ptr + new_shape_length);
  auto& x_info = backend::get_tensor_info(x_id);
  auto& out_info = backend::get_tensor_info_out(out_id);

  switch (dtype) {
    case DType::float32:
      tile_slow<float>(x_info.f32(), x_shape, new_shape, out_info.f32_write());
      break;
    case DType::int32:
      tile_slow<int32_t>(x_info.i32(), x_shape, new_shape,
                         out_info.i32_write());
      break;
    case DType::boolean:
      tile_slow<bool>(x_info.b(), x_shape, new_shape, out_info.b_write());
      break;
    default:
      util::warn("Tile for tensor id %d failed. Unknown dtype %d", x_id, dtype);
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
