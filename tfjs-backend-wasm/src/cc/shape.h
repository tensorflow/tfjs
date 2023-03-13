/* Copyright 2023 Google LLC.
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

#ifndef SHAPE_H_
#define SHAPE_H_

#include <array>
#include <limits>
#include <utility>
#include <vector>

namespace tfjs {

// The shape of a tensor.
template <typename T, int N>
class Shape {
 public:
  explicit Shape(std::array<T, N> data) : data_(std::move(data)) {
    static_assert(std::numeric_limits<T>::is_integer,
                  "Tensor shape must be a list of integers.");
    static_assert(N > 0, "Tensor shape must not have 0 length");
  }
  explicit Shape(const T* data_ptr)
      : Shape(BuildShapeFromData(data_ptr, std::make_index_sequence<N>{})) {}

  // Returns the size of the tensor with this shape.
  inline T size() const {
    T size = 1;
    for (size_t i = 0; i < N; ++i) {
      size *= data_[i];
    }
    return size;
  }

  // Returns the flat offset of a N-D tensor given the indices.
  inline T offset(const std::array<T, N>& indices) const {
    T offset = 0;
    for (size_t i = 0; i < N - 1; ++i) {
      offset += indices[i];
      offset *= data_[i + 1];
    }
    offset += indices[N - 1];
    return offset;
  }

  inline T& operator[](size_t i) { return data_[i]; }
  inline const T& operator[](size_t i) const { return data_[i]; }

  inline T& at(size_t i) { return data_.at(i); }
  inline const T& at(size_t i) const { return data_.at(i); }

  const std::array<T, N>& array() const { return data_; }

 private:
  // Internal constructor helpers for converting data into std::array.
  template <typename Array, size_t... I>
  inline static Shape BuildShapeFromData(const Array& a,
                                         std::index_sequence<I...>) {
    return Shape({a[I]...});
  }

  std::array<T, N> data_;
};

}  // namespace tfjs

#endif
