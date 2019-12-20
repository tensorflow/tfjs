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

#include <cstddef>
#include <vector>

#include "src/cc/scatter_impl.h"

#include "src/cc/backend.h"
#include "src/cc/util.h"

namespace tfjs {
namespace wasm {
void scatter(const int* indices_ptr, const float* updates_ptr,
             size_t slice_rank, size_t num_updates, size_t slice_size,
             const std::vector<size_t>& strides_ptr,
             const std::vector<size_t>& shape_ptr, float* out_buf_ptr) {
  for (size_t i = 0; i < num_updates; ++i) {
    size_t flattened_index = 0;
    for (size_t j = 0; j < slice_rank; ++j) {
      int dim = indices_ptr[i * slice_rank + j];
      flattened_index += dim * strides_ptr[j];
    }

    for (size_t k = 0; k < slice_size; ++k) {
      out_buf_ptr[flattened_index * slice_size + k] =
          updates_ptr[i * slice_size + k];
    }
  }
}
}  // namespace wasm
}  // namespace tfjs
