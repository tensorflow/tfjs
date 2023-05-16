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
  return std::fmod(std::fmod(x, 2 * M_PI) + 2 * M_PI, 2 * M_PI);
}

template <typename T, bool is_shifted = false>
inline T SinFixedImpl(T x) {
  if (std::isnan(x)) return x;
  if (!is_shifted) x = ShiftRadianToZeroTo2PI(x);

  if (x < M_PI_4) {
    return std::sin(x);
  } else if (x < M_PI_2) {
    return std::cos(M_PI_2 - x);
  } else if (x < M_PI) {
    return SinFixedImpl<T, /*is_shifted=*/true>(M_PI - x);
  } else {
    return -SinFixedImpl<T, /*is_shifted=*/true>(2 * M_PI - x);
  }
}

template <typename T, bool is_shifted = false>
inline T CosFixedImpl(T x) {
  if (std::isnan(x)) return x;
  if (!is_shifted) x = ShiftRadianToZeroTo2PI(x);

  if (x < M_PI_4) {
    return std::cos(x);
  } else if (x < M_PI_2) {
    return std::sin(M_PI_2 - x);
  } else if (x < M_PI) {
    return -CosFixedImpl<T, /*is_shifted=*/true>(M_PI - x);
  } else {
    return CosFixedImpl<T, /*is_shifted=*/true>(2 * M_PI - x);
  }
}

}  // namespace

float SinFixed(float x) { return SinFixedImpl(x); }

float CosFixed(float x) { return CosFixedImpl(x); }

float TanFixed(float x) {
  // TODO: Check if this work on iOS 11/12.
  return std::tan(x);
}

}  // namespace sin_cos_workaround
}  // namespace tfjs
