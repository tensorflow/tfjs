/* Copyright 2020 Google LLC. All Rights Reserved.
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

#ifndef KERNELS_PADV2_H_
#define KERNELS_PADV2_H_

#include <cstddef>

#include "src/cc/backend.h"

namespace tfjs {
namespace wasm {
extern "C" {

void PadV2(const size_t x_id, const size_t* x_shape_ptr,
           const size_t x_shape_length, const DType dtype,
           const size_t* pre_paddings_ptr, const size_t* post_paddings_ptr,
           const float pad_value, const size_t out_id);
}

}  // namespace wasm
}  // namespace tfjs

#endif  // KERNELS_PADV2_H_
