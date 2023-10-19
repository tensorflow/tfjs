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

namespace tfjs {
namespace wasm {

template <typename T>
void scatter(const int* indices_ptr, const T* updates_ptr,
             const size_t slice_rank, const size_t num_updates,
             const size_t slice_size, const std::vector<size_t>& strides_ptr,
             const size_t output_size, const T default_value,
             const bool sum_dupe_indices, const bool update_as_scalar,
             T* out_buf_ptr);

}  // namespace wasm
}  // namespace tfjs

#endif  // SCATTER_IMPL_H_
