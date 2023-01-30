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

#ifndef SCATTER_IMPL_H_
#define SCATTER_IMPL_H_

#include <cstddef>
#include <vector>

#include "tfjs-backend-wasm/src/cc/scatter_impl.h"

namespace tfjs {
namespace wasm {

template <typename T>
void scatter(const int* indices_ptr, const T* updates_ptr,
             const size_t slice_rank, const size_t num_updates,
             const size_t slice_size, const std::vector<size_t>& strides_ptr,
             const size_t output_size, const T default_value,
             const bool sum_dupe_indices, const bool update_as_scalar,
             T* out_buf_ptr) {
  std::fill(out_buf_ptr, out_buf_ptr + output_size, default_value);

  for (size_t i = 0; i < num_updates; ++i) {
    size_t flattened_index = 0;
    for (size_t j = 0; j < slice_rank; ++j) {
      flattened_index += *indices_ptr * strides_ptr[j];

      indices_ptr++;
    }

    T* out = out_buf_ptr + flattened_index * slice_size;

    for (size_t k = 0; k < slice_size; ++k) {
      if (sum_dupe_indices) {
        *out += *updates_ptr;
      } else {
        *out = *updates_ptr;
      }

      out++;
      if (!update_as_scalar) {
        updates_ptr++;
      }
    }
  }
}

// Templates need explicit instantiation when implemented in a .cc file.
template void scatter<float>(const int* indices_ptr, const float* updates_ptr,
                             const size_t slice_rank, const size_t num_updates,
                             const size_t slice_size,
                             const std::vector<size_t>& strides_ptr,
                             const size_t output_size,
                             const float default_value,
                             const bool sum_dupe_indices,
                             const bool update_as_scalar, float* out_buf_ptr);
template void scatter<int32_t>(
    const int* indices_ptr, const int32_t* updates_ptr, const size_t slice_rank,
    const size_t num_updates, const size_t slice_size,
    const std::vector<size_t>& strides_ptr, const size_t output_size,
    const int32_t default_value, const bool sum_dupe_indices,
    const bool update_as_scalar, int32_t* out_buf_ptr);
template void scatter<bool>(const int* indices_ptr, const bool* updates_ptr,
                            const size_t slice_rank, const size_t num_updates,
                            const size_t slice_size,
                            const std::vector<size_t>& strides_ptr,
                            const size_t output_size, const bool default_value,
                            const bool sum_dupe_indices,
                            const bool update_as_scalar, bool* out_buf_ptr);

}  // namespace wasm
}  // namespace tfjs

#endif  // SCATTER_IMPL_H_
