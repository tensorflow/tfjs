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

#ifndef TRANSPOSE_IMPL_H_
#define TRANSPOSE_IMPL_H_

#include <cstddef>
#include <vector>

namespace tfjs {
namespace wasm {

template <typename T>
void transpose(const T* x_data, const std::vector<size_t>& x_shape,
               const std::vector<size_t>& perm, T* out_data);

}  // namespace wasm
}  // namespace tfjs
#endif  // TRANSPOSE_IMPL_H_
