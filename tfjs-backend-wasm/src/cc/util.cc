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

#include <vector>

#include "src/cc/util.h"

namespace tfjs {
namespace util {

std::vector<int> compute_strides(const std::vector<int> shape) {
  int rank = shape.size();
  std::vector<int> strides(rank - 1);
  // Last dimension has implicit stride of 1, thus having D-1 (instead of D)
  // strides.
  strides[rank - 2] = shape[rank - 1];
  for (int i = rank - 3; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

}  // namespace util
}  // namespace tfjs
