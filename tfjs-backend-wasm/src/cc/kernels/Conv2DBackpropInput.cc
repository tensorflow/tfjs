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

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include <vector>

#include "src/cc/backend.h"
#include "src/cc/util.h"

namespace tfjs {
namespace wasm {
extern "C" {
#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void Conv2DBackpropInput(
    const size_t dy_id, const size_t filter_id, const size_t batch_size,
    const size_t filter_height, const size_t filter_width,
    const size_t in_height, const size_t in_width, const size_t in_channels,
    const size_t out_height, const size_t out_width, const size_t out_channels,
    const size_t stride_height, const size_t stride_width, const size_t top_pad,
    const size_t left_pad, const size_t flt_s0, const size_t flt_s1,
    const size_t flt_s2, const size_t x_batch_stride, const size_t x_row_stride,
    const size_t x_col_stride, const size_t x_channel_stride,
    const size_t y_batch_stride, const size_t y_row_stride,
    const size_t y_col_stride, const size_t y_channel_stride,
    const size_t out_id) {}
}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
