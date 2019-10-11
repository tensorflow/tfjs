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

#include <vector>

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

namespace tfjs {
namespace backend {
// Returns the tensor information object associated with a given tensor_id
// bucket.
TensorInfo get_tensor_info(int tensor_id);

// Registers a function callback to be called when a tensor with a given ID is
// disposed.
typedef void (*DisposeFunction)(int);
void register_disposal_callback(int tensor_id, DisposeFunction dispose_fn);

// Returns the number of tensors registered and owned by the backend.
int num_tensors();

// The number of instantiated XNN operators.
extern int xnn_operator_count;
}  // namespace backend

namespace wasm {
extern "C" {
// Initializes the WASM backend.
void init();
// Registers a tensor with a tensor ID, shape information, dtype, and the
// pointer to where the tensor data lives.
void register_tensor(int tensor_id, int *shape_ptr, int shape_length,
                     DType dtype, void *memory_offset);
// Disposes the internal bookeeping for a given tensor ID.
void dispose_data(int tensor_id);
// Disposes all internal state.
void dispose();
}

}  // namespace wasm
}  // namespace tfjs

#endif  // TFJS_BACKEND_H
