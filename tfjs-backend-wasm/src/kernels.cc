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

#include "kernels.h"

#include <algorithm>

namespace tfjs {
namespace kernels {

// TODO(smilkov): Consider inlining small methods.

template <class T>
void add(T* a_buf, int a_size, T* b_buf, int b_size, T* out_buf) {
  int size = std::max(a_size, b_size);
  for (int i = 0; i < size; ++i) {
    out_buf[i] = a_buf[i % a_size] + b_buf[i % b_size];
  }
}

// Templates need explicit instantiation when implemented in a .cc file.
template void add<float>(float* a_buf, int a_size, float* b_buf, int b_size,
                         float* out_buf);
template void add<int>(int* a_buf, int a_size, int* b_buf, int b_size,
                       int* out_buf);
template void add<bool>(bool* a_buf, int a_size, bool* b_buf, int b_size,
                        bool* out_buf);

}  // namespace kernels
}  // namespace tfjs
