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

#ifndef KERNELS_DEPTHWISECONV2DNATIVE_H_
#define KERNELS_DEPTHWISECONV2DNATIVE_H_

namespace tfjs {

namespace wasm {
extern "C" {
void DepthwiseConv2dNative(const int x_id, const int batch_size,
                           const int input_height, const int input_width,
                           const int filter_id, const int filter_height,
                           const int filter_width, int pad_top, int pad_right,
                           int pad_bottom, int pad_left, const int is_same_pad,
                           const int dilation_height, const int dilation_width,
                           const int stride_height, const int stride_width,
                           const int input_channels, const int output_channels,
                           const int out_id);
}

}  // namespace wasm
}  // namespace tfjs

#endif  // KERNELS_DEPTHWISECONV2DNATIVE_H_
