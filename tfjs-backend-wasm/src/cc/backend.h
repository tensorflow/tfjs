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

#ifndef TFJS_BACKEND_H
#define TFJS_BACKEND_H

namespace tfjs {

enum DType {
  float32 = 0,
  int32 = 1,
  boolean = 2,
};

// A union of pointers that points to memory for a given tensor.
union DataPtrUnion {
  float *f32;
  int *i32;
  bool *b;
};

// Holds information about a tensor such as dtype, shape and pointer to its data
// in memory.
struct TensorInfo {
  // Pointer to the bytes where the data is allocated.
  DataPtrUnion buf;
  DType dtype;
  std::vector<int> shape;
  // Total number of elements.
  int size;
};

TensorInfo get_tensor_info(int tensor_id);

}  // namespace tfjs

#endif  // TFJS_BACKEND_H
