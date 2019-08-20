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

#include <emscripten.h>
#include <math.h>
#include <cstdio>
#include <map>
#include <vector>

#include "kernels.h"
#include "util.h"

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

// Maps a unique tensor id to info about that tensor. The map owns all of its
// entries.
std::map<int, TensorInfo> data;

// We use C-style API to interface with Javascript.
extern "C" {

EMSCRIPTEN_KEEPALIVE
void register_tensor(int data_id, int *shape_ptr, int shape_length, DType dtype,
                     void *memory_offset) {
  auto shape = std::vector<int>(shape_ptr, shape_ptr + shape_length);
  auto size = util::size_from_shape(shape);

  TensorInfo info = {{}, dtype, std::move(shape), size};
  switch (dtype) {
    case DType::float32:
      info.buf.f32 = static_cast<float *>(memory_offset);
      break;
    case DType::int32:
      info.buf.i32 = static_cast<int *>(memory_offset);
      break;
    case DType::boolean:
      info.buf.b = static_cast<bool *>(memory_offset);
      break;
    default:
      util::warn("Failed to register tensor id %d failed. Unknown dtype %d",
                 data_id, dtype);
  }
  // We move info to avoid a copy.
  data.insert({data_id, std::move(info)});
}

EMSCRIPTEN_KEEPALIVE
void dispose_data(int data_id) {
  TensorInfo info = data.at(data_id);
  switch (info.dtype) {
    case DType::float32:
      free(info.buf.f32);
      break;
    case DType::int32:
      free(info.buf.i32);
      break;
    case DType::boolean:
      free(info.buf.b);
      break;
    default:
      util::warn("Dispose for tensor id %d failed. Unknown dtype %d", data_id,
                 info.dtype);
  }
  data.erase(data_id);
}

EMSCRIPTEN_KEEPALIVE
void add(int a_id, int b_id, int out_id) {
  const auto a_info = data.at(a_id);
  const auto b_info = data.at(b_id);
  const auto out_info = data.at(out_id);
  switch (a_info.dtype) {
    case DType::float32:
      kernels::add(a_info.buf.f32, a_info.size, b_info.buf.f32, b_info.size,
                   out_info.buf.f32);
      break;
    case DType::int32:
      kernels::add(a_info.buf.i32, a_info.size, b_info.buf.i32, b_info.size,
                   out_info.buf.i32);
      break;
    case DType::boolean:
      kernels::add(a_info.buf.b, a_info.size, b_info.buf.b, b_info.size,
                   out_info.buf.b);
      break;
    default:
      util::warn("Add for tensor ids %d and %d failed. Unknown dtype %d", a_id,
                 b_id, a_info.dtype);
  }
}

EMSCRIPTEN_KEEPALIVE
void dispose() {
  for (auto const &element : data) {
    dispose_data(element.first);
  }
  data.clear();
}

}  // extern "C"
}  // namespace tfjs
