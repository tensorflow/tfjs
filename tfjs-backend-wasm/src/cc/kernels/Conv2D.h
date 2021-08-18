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

#ifndef KERNELS_CONV2D_H_
#define KERNELS_CONV2D_H_

#include <cstddef>

#include "tfjs-backend-wasm/src/cc/conv2d_impl.h"

namespace tfjs {
namespace wasm {
extern "C" {

void Conv2D(const size_t x_id, const size_t batch_size,
            const size_t input_height, const size_t input_width,
            const size_t filter_id, const size_t filter_height,
            const size_t filter_width, size_t pad_top, size_t pad_right,
            size_t pad_bottom, size_t pad_left, const size_t is_same_pad,
            const size_t dilation_height, const size_t dilation_width,
            const size_t stride_height, const size_t stride_width,
            const size_t input_channels, const size_t output_channels,
            const size_t out_id);
}

}  // namespace wasm
}  // namespace tfjs

#endif  // KERNELS_CONV2D_H_
