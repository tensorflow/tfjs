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

#ifndef TFJS_WASM_UTIL_H_
#define TFJS_WASM_UTIL_H_

#include <string.h>
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
  for (auto const& value : v) {
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

}  // namespace util
}  // namespace tfjs
#endif  // TFJS_WASM_UTIL_H_
