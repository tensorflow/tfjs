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

#ifndef FFT_IMPL_H_
#define FFT_IMPL_H_

#include <cstddef>

#include "src/cc/backend.h"

namespace tfjs {
namespace wasm {

void fft(const size_t real_input_id, const size_t imag_input_id,
         const size_t outer_dim, const size_t inner_dim,
         const size_t is_real_component, const size_t out_id);
}  // namespace wasm
}  // namespace tfjs

#endif  // FFT_IMPL_H_
