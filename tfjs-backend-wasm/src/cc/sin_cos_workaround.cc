/* Copyright 2021 Google LLC. All Rights Reserved.
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

#include <cmath>

#include "tfjs-backend-wasm/src/cc/sin_cos_workaround.h"

namespace tfjs {
namespace sin_cos_workaround {

namespace {

template <typename T>
inline T ShiftRadianToZeroTo2PI(const T& x) {
  if (std::isnan(x)) {
    return x;
  }
  return std::fmod(std::fmod(x, 2 * M_PI) + 2 * M_PI, 2 * M_PI);
}

template <typename T>
inline T SinZeroTo2PI(const T& x) {
  if (std::isnan(x)) {
    return x;
  }

  if (x < M_PI_4) {
    return std::sin(x);
  } else if (x < M_PI_2) {
    return std::cos(M_PI_2 - x);
  } else if (x < M_PI) {
    return SinZeroTo2PI<T>(M_PI - x);
  } else {
    return -SinZeroTo2PI<T>(2 * M_PI - x);
  }
}

template <typename T>
inline T CosZeroTo2PI(const T& x) {
  if (std::isnan(x)) {
    return x;
  }

  if (x < M_PI_4) {
    return std::cos(x);
  } else if (x < M_PI_2) {
    return std::sin(M_PI_2 - x);
  } else if (x < M_PI) {
    return -CosZeroTo2PI<T>(M_PI - x);
  } else {
    return CosZeroTo2PI<T>(2 * M_PI - x);
  }
}

}  // namespace

float SinFixed(float x) { return SinZeroTo2PI(ShiftRadianToZeroTo2PI(x)); }

float CosFixed(float x) { return CosZeroTo2PI(ShiftRadianToZeroTo2PI(x)); }

float TanFixed(float x) { return std::tan(x); }

}  // namespace sin_cos_workaround
}  // namespace tfjs
