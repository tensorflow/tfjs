/* Copyright 2019 Google LLC. All Rights Reserved.
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

#include "tfjs-backend-wasm/src/cc/util.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <vector>

namespace tfjs {
namespace util {
namespace {

TEST(util, offset_2d) {
  std::array<size_t, 2> coord = {0, 0};
  std::array<size_t, 1> stride = {0};
  EXPECT_EQ(0, offset(coord[0], coord[1], stride[0]));

  coord = {2, 3};
  stride = {5};
  EXPECT_EQ(13, offset(coord[0], coord[1], stride[0]));
}

TEST(util, offset_3d) {
  std::array<size_t, 3> coord = {0, 0, 0};
  std::array<size_t, 2> stride = {0, 0};
  EXPECT_EQ(0, offset(coord[0], coord[1], coord[2], stride[0], stride[1]));

  coord = {3, 5, 7};
  stride = {4, 3};
  EXPECT_EQ(34, offset(coord[0], coord[1], coord[2], stride[0], stride[1]));
}

TEST(util, offset_4d) {
  std::array<size_t, 4> coord = {0, 0, 0, 0};
  std::array<size_t, 3> stride = {0, 0, 0};
  EXPECT_EQ(0, offset(coord[0], coord[1], coord[2], coord[3], stride[0],
                      stride[1], stride[2]));

  coord = {1, 2, 3, 4};
  stride = {5, 7, 9};
  EXPECT_EQ(50, offset(coord[0], coord[1], coord[2], coord[3], stride[0],
                       stride[1], stride[2]));
}

TEST(util, offset_5d) {
  std::array<size_t, 5> coord = {0, 0, 0, 0, 0};
  std::array<size_t, 4> stride = {0, 0, 0, 0};
  EXPECT_EQ(0, offset(coord[0], coord[1], coord[2], coord[3], coord[4],
                      stride[0], stride[1], stride[2], stride[3]));

  coord = {1, 2, 3, 4, 5};
  stride = {5, 7, 9, 11};
  EXPECT_EQ(95, offset(coord[0], coord[1], coord[2], coord[3], coord[4],
                       stride[0], stride[1], stride[2], stride[3]));
}

TEST(util, size_from_shape) {
  std::vector<size_t> shape = {};
  EXPECT_EQ(1, size_from_shape(shape));

  shape = {3};
  EXPECT_EQ(3, size_from_shape(shape));

  shape = {3, 4};
  EXPECT_EQ(12, size_from_shape(shape));

  shape = {1, 3, 5};
  EXPECT_EQ(15, size_from_shape(shape));

  shape = {2, 3, 4};
  EXPECT_EQ(24, size_from_shape(shape));

  shape = {2, 3, 4, 5};
  EXPECT_EQ(120, size_from_shape(shape));
}

TEST(util, loc_to_offset) {
  std::vector<size_t> loc = {};
  std::vector<size_t> strides = {};
  EXPECT_EQ(0, loc_to_offset(loc, strides));

  loc = {5};
  strides = {};
  EXPECT_EQ(5, loc_to_offset(loc, strides));

  loc = {3, 5};
  strides = {7};
  EXPECT_EQ(26, loc_to_offset(loc, strides));

  loc = {6, 0, 3};
  strides = {8, 4};
  EXPECT_EQ(51, loc_to_offset(loc, strides));

  loc = {8, 0, 1, 1};
  strides = {8, 4, 2};
  EXPECT_EQ(67, loc_to_offset(loc, strides));
}

TEST(util, offset_to_loc) {
  size_t offset = 5;
  std::vector<size_t> strides = {};
  EXPECT_EQ(std::vector<size_t>({5}), offset_to_loc(offset, strides));

  offset = 26;
  strides = {7};
  EXPECT_EQ(std::vector<size_t>({3, 5}), offset_to_loc(offset, strides));

  offset = 51;
  strides = {8, 4};  // shape is [7, 2, 4]
  EXPECT_EQ(std::vector<size_t>({6, 0, 3}), offset_to_loc(offset, strides));

  offset = 67;
  strides = {8, 4, 2};  // shape is [9, 2, 2, 2]
  EXPECT_EQ(std::vector<size_t>({8, 0, 1, 1}), offset_to_loc(offset, strides));
}

TEST(util, compute_strides) {
  std::vector<size_t> shape = {5, 7};
  EXPECT_EQ(std::vector<size_t>({7}), compute_strides(shape));

  shape = {3, 5, 7};
  EXPECT_EQ(std::vector<size_t>({35, 7}), compute_strides(shape));

  shape = {3, 5, 7, 9};
  EXPECT_EQ(std::vector<size_t>({315, 63, 9}), compute_strides(shape));

  shape = {2, 3, 5, 7, 9};
  EXPECT_EQ(std::vector<size_t>({945, 315, 63, 9}), compute_strides(shape));

  shape = {2, 2, 2, 2, 2, 2};
  EXPECT_EQ(std::vector<size_t>({32, 16, 8, 4, 2}), compute_strides(shape));
}

}  // namespace
}  // namespace util
}  // namespace tfjs
