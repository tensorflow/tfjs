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

#include "src/cc/kernels.h"

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

const int blockSize = 48;
void batchMatMul(float* a_buf, float* b_buf, int sharedDim, int leftDim,
                 int rightDim, int batchDim, int aBatch, int aOuterStep,
                 int aInnerStep, int bBatch, int bOuterStep, int bInnerStep,
                 float* out_buf) {
  int size = leftDim * rightDim;

  // Zero out the output buffer because it might have been used before.
  std::fill(out_buf, out_buf + size, 0);

  for (int b = 0; b < batchDim; b++) {
    for (int i0 = 0; i0 < leftDim; i0 += blockSize) {
      for (int j0 = 0; j0 < rightDim; j0 += blockSize) {
        for (int k0 = 0; k0 < sharedDim; k0 += blockSize) {
          // for when blockSize doesn't evenly divide the input
          int iBlock = std::min(i0 + blockSize, leftDim);
          int jBlock = std::min(j0 + blockSize, rightDim);
          int kBlock = std::min(k0 + blockSize, sharedDim);

          for (int i = i0; i < iBlock; i++) {
            for (int j = j0; j < jBlock; j++) {
              float sum = 0.0;

              for (int k = k0; k < kBlock; k++) {
                sum += a_buf[b * aBatch + i * aOuterStep + k * aInnerStep] *
                       b_buf[k * bInnerStep + j * bOuterStep + b * bBatch];
              }
              out_buf[b * size + (i * rightDim + j)] += sum;
            }
          }
        }
      }
    }
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
