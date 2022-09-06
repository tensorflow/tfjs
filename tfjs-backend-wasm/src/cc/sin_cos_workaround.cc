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
#include <math.h>

#include "tfjs-backend-wasm/src/cc/sin_cos_workaround.h"

namespace tfjs {
namespace sin_cos_workaround {

float sin_fixed(float x) {
  if (isnan(x)) return nan("");
  auto zero_to_2pi = fmod(fmod(x, 2 * M_PI) + 2 * M_PI, 2 * M_PI);

  if (zero_to_2pi < M_PI_4) {
    return sin(zero_to_2pi);
  } else if (zero_to_2pi < M_PI_2) {
    auto past_pi_4 = zero_to_2pi - M_PI_4;
    return cos(M_PI_4 - past_pi_4);
  } else if (zero_to_2pi < M_PI) {
    auto past_pi_2 = zero_to_2pi - M_PI_2;
    return sin_fixed(M_PI_2 - past_pi_2);
  } else {
    return -sin_fixed(2 * M_PI - zero_to_2pi);
  }
}

float cos_fixed(float x) { return sin_fixed(x + M_PI_2); }

float tan_fixed(float x) {
  if (isnan(x)) return nan("");
  return sin_fixed(x) / cos_fixed(x);
}

}  // namespace sin_cos_workaround
}  // namespace tfjs
