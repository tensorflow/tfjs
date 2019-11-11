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

#ifndef BACKEND_H_
#define BACKEND_H_

// This enum should align with the DType defined in kernels/types.ts.
enum DType {
  float32 = 0,
  int32 = 1,
  boolean = 2,
  str = 3,
  complex64 = 4,
};

// Holds the memory offset and the size of a tensor.
struct TensorInfo {
  // Pointer to the bytes where the data is allocated.
  void *memory_offset;
  // Total number of elements.
  const int size;

  const float *f32() const {
    return reinterpret_cast<const float *>(memory_offset);
  }

  float *f32_write() { return reinterpret_cast<float *>(memory_offset); }

  const int *i32() const {
    return reinterpret_cast<const int *>(memory_offset);
  }

  int *i32_write() { return reinterpret_cast<int *>(memory_offset); }

  const bool *b() const {
    return reinterpret_cast<const bool *>(memory_offset);
  }

  bool *b_write() { return reinterpret_cast<bool *>(memory_offset); }

  // Delete the copy constructor to avoid copying.
  TensorInfo(const TensorInfo &) = delete;
  void operator=(const TensorInfo &) = delete;

  // Bring back the move constructor.
  TensorInfo(TensorInfo &&) = default;
};

namespace tfjs {
namespace backend {
// Returns the tensor information object associated with a given tensor_id
// bucket.
const TensorInfo &get_tensor_info(int tensor_id);
// Same as above, but gives write access to the tensor info.
TensorInfo &get_tensor_info_out(int tensor_id);

// Registers a function callback to be called when a tensor with a given ID is
// disposed.
typedef void (*DisposeFunction)(int);
void register_disposal_callback(int tensor_id, DisposeFunction dispose_fn);

// Returns the number of tensors registered and owned by the backend.
const int num_tensors();

// Throws a JavaScript exception.
void throw_js_exception(const char *format, ...);
void set_throw_js_exception_fn(void (*f)(char *, int));

// The number of instantiated XNN operators.
extern int xnn_operator_count;
}  // namespace backend

namespace wasm {
extern "C" {
// Initializes the WASM backend.
void init(int fp);

// Registers a tensor with a tensor ID, size, and the pointer to where the
// tensor data lives.
void register_tensor(const int tensor_id, const int size, void *memory_offset);

// Disposes the internal bookeeping for a given tensor ID.
void dispose_data(const int tensor_id);

// Disposes all internal state.
void dispose();
}

}  // namespace wasm
}  // namespace tfjs

#endif  // BACKEND_H_
