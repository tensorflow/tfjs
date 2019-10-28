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

#include <vector>

namespace tfjs {
namespace util {

// Logs and flushes the message to the js console (console.log).
void log(const char* format, ...);

// Logs and flushes the message to the js console (console.err).
void warn(const char* format, ...);

// Helper method to log values in a vector. Used for debugging.
template <class T>
void log_vector(const std::vector<T>& v);

// Returns the size of the vector, given its shape.
int size_from_shape(const std::vector<int>& shape);

std::vector<int> index_to_loc(int index, const std::vector<int>& strides);

int loc_to_index(const std::vector<int>& loc, const std::vector<int>& strides);

std::vector<int> compute_strides(const std::vector<int> shape);

}  // namespace util
}  // namespace tfjs
#endif  // UTIL_H_
