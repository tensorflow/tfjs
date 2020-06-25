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

#ifndef KERNELS_RESIZEBILINEAR_H_
#define KERNELS_RESIZEBILINEAR_H_

#include <cstddef>

namespace tfjs {
namespace wasm {
extern "C" {

void ResizeBilinear(size_t x_id, size_t batch, size_t old_height,
                    size_t old_width, size_t num_channels, size_t new_height,
                    size_t new_width, bool align_corners, size_t out_id);
}

}  // namespace wasm
}  // namespace tfjs

#endif  // KERNELS_RESIZEBILINEAR_H_
