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

#include <cstddef>
#include <vector>

#include "src/cc/util.h"

namespace tfjs {
namespace util {

const std::vector<size_t> compute_strides(const std::vector<size_t> shape) {
  const size_t rank = shape.size();
  std::vector<size_t> strides(rank - 1);

  if (rank < 2) {
    return strides;
  }

  // Last dimension has implicit stride of 1, thus having D-1 (instead of D)
  // strides.
  strides[rank - 2] = shape[rank - 1];

  if (rank < 3) {
    return strides;
  }

  // We do i < rank here because i <= 0 is always true for unsigned integers and
  // decrementing will wrap to the max int.
  for (size_t i = rank - 3; i < rank; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  return strides;
}
}  // namespace util
}  // namespace tfjs
