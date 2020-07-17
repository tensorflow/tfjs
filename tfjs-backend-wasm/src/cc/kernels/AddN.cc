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

#include "src/cc/backend.h"
#include "src/cc/util.h"

namespace {

template <typename T>
void addn(const std::vector<const T*>& inputs_buf, const size_t size,
          T* out_buf) {
  // Initialize the output to 0.
  memset(out_buf, 0, size * sizeof(T));

  for (size_t in_idx = 0; in_idx < inputs_buf.size(); ++in_idx) {
    const T* input = inputs_buf[in_idx];
    for (size_t i = 0; i < size; ++i) {
      out_buf[i] += input[i];
    }
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
void AddN(const size_t* input_ids_ptr, const size_t input_ids_len,
          const DType dtype, const size_t out_id) {
  std::vector<size_t> inputs(input_ids_ptr, input_ids_ptr + input_ids_len);
  auto& out_info = backend::get_tensor_info_out(out_id);
  std::vector<void*> inputs_buf;
  std::transform(
      inputs.begin(), inputs.end(), std::back_inserter(inputs_buf),
      [](size_t id) { return backend::get_tensor_info(id).memory_offset; });

  switch (dtype) {
    case DType::float32:
      addn<float>(reinterpret_cast<std::vector<const float*>&>(inputs_buf),
                  out_info.size, out_info.f32_write());
      break;
    case DType::int32:
      addn<int32_t>(reinterpret_cast<std::vector<const int*>&>(inputs_buf),
                    out_info.size, out_info.i32_write());
      break;
    case DType::boolean:
      addn<bool>(reinterpret_cast<std::vector<const bool*>&>(inputs_buf),
                 out_info.size, out_info.b_write());
      break;
    default:
      util::warn("AddN failed. Unknown dtype %d", dtype);
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
