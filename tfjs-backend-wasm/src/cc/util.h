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

#ifndef UTIL_H_
#define UTIL_H_

#include <cstdarg>
#include <cstdio>
#include <vector>

namespace tfjs {
namespace util {

inline void print_log(const char* format, va_list args) {
  // TODO(smilkov): Bind directly to `console.log` instead of depending on
  // stdio, to reduce wasm binary size.
  vfprintf(stdout, format, args);
}

inline void print_warn(const char* format, va_list args) {
  // TODO(smilkov): Bind directly to `console.warn` instead of depending on
  // stdio, to reduce wasm binary size.
  vfprintf(stderr, format, args);
}

// Logs the message to the console without flushing.
inline void print_log(const char* format, ...) {
  va_list args;
  va_start(args, format);
  print_log(format, args);
}

inline void print_warn(const char* format, ...) {
  va_list args;
  va_start(args, format);
  print_warn(format, args);
}

// Logs and flushes the message to the js console (console.log).
inline void log(const char* format, ...) {
  va_list args;
  va_start(args, format);
  print_log(format, args);
  print_log("\n");
}

// Logs and flushes the message to the js console (console.err).
inline void warn(const char* format, ...) {
  va_list args;
  va_start(args, format);
  print_warn(format, args);
  print_warn("\n");
}

// Helper method to log values in a vector. Used for debugging.
template <class T>
inline void log_vector(const std::vector<T>& v) {
  print_log("[", 0);
  for (const auto& value : v) {
    print_log("%d,", value);
  }
  print_log("]\n", 0);
}

// Returns the size of the vector, given its shape.
inline int size_from_shape(const std::vector<int>& shape) {
  int prod = 1;
  for (const auto& v : shape) {
    prod *= v;
  }
  return prod;
}

// Returns the indices of an n-dim tensor given the flat offset and its strides.
inline const std::vector<int> offset_to_loc(int index,
                                            const std::vector<int>& strides) {
  int rank = strides.size() + 1;
  std::vector<int> loc(rank);
  if (rank == 0) {
    return loc;
  } else if (rank == 1) {
    loc[0] = index;
    return loc;
  }
  for (int i = 0; i < rank - 1; ++i) {
    int stride = strides[i];
    loc[i] = index / stride;
    index -= loc[i] * stride;
  }
  loc[rank - 1] = index;
  return loc;
}

// Returns the flat offset of an n-dim tensor given the indices and strides.
inline int loc_to_offset(const std::vector<int>& loc,
                         const std::vector<int>& strides) {
  size_t rank = loc.size();
  if (rank == 0) {
    return 0;
  } else if (rank == 1) {
    return loc[0];
  }
  size_t index = loc[loc.size() - 1];
  for (size_t i = 0; i < loc.size() - 1; ++i) {
    index += strides[i] * loc[i];
  }
  return index;
}

// Returns the flat offset of a 2D tensor given the indices and the stride.
inline int offset(int i1, int i2, int s1) { return i1 * s1 + i2; }

// Returns the flat offset of a 3D tensor given the indices and the strides.
inline int offset(int i1, int i2, int i3, int s1, int s2) {
  return i1 * s1 + i2 * s2 + i3;
}

// Returns the flat offset of a 4D tensor given the indices and the strides.
inline int offset(int i1, int i2, int i3, int i4, int s1, int s2, int s3) {
  return i1 * s1 + i2 * s2 + i3 * s3 + i4;
}

// Returns the flat offset of a 5D tensor given the indices and the strides.
inline int offset(int i1, int i2, int i3, int i4, int i5, int s1, int s2,
                  int s3, int s4) {
  return i1 * s1 + i2 * s2 + i3 * s3 + i4 * s4 + i5;
}

// Returns the strides of a tensor given its shape. Note that the strides
// are of length R-1 where R is the rank of the tensor.
const std::vector<int> compute_strides(const std::vector<int> shape);

}  // namespace util
}  // namespace tfjs
#endif  // UTIL_H_
