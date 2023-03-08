/* Copyright 2023 Google LLC.
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

#ifndef CONV3D_H_
#define CONV3D_H_

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include <cstddef>
#include "tfjs-backend-wasm/src/cc/shape.h"

namespace tfjs::wasm {

namespace {

inline int AddUntilNonNegative(int v, int d) {
  if (v >= 0) {
    return v;
  }
  return (v % d + d) % d;
}

}  // namespace

struct NDHWCConv3DInfo {
  int batch_size;
  // Since Pool3D ops (AvgPool3D and MaxPool3D) support 3D filter only, in
  // channels should always equal to out channels.
  int channel_size;
  int in_depth;
  int in_height;
  int in_width;
  int out_depth;
  int out_height;
  int out_width;

  int stride_depth;
  int stride_height;
  int stride_width;
  int dilation_depth;
  int dilation_height;
  int dilation_width;
  int filter_depth;
  int filter_height;
  int filter_width;
  int pad_front;
  int pad_top;
  int pad_left;

  inline Shape<int, 5> in_shape() const {
    return Shape<int, 5>(
        {batch_size, in_depth, in_height, in_width, channel_size});
  }
  inline Shape<int, 5> out_shape() const {
    return Shape<int, 5>(
        {batch_size, out_depth, out_height, out_width, channel_size});
  }

  inline int in_offset(int b, int d, int h, int w, int c) const {
    return in_shape().offset({b, d, h, w, c});
  }
  inline int out_offset(int b, int d, int h, int w, int c) const {
    return out_shape().offset({b, d, h, w, c});
  }
  inline int int_size() const { return in_shape().size(); }
  inline int out_size() const { return out_shape().size(); }
};
template <typename IN, typename OUT>
inline void NDHWCConv3DImpl(const IN* x_buf, OUT* out_buf,
                            const NDHWCConv3DInfo& info) {
  for (int b = 0; b < info.batch_size; ++b) {
    for (int yf = 0; yf < info.out_depth; ++yf) {
      int xf_corner = yf * info.stride_depth - info.pad_front;
      for (int wf = 0; wf < info.filter_depth; ++wf) {
        int xf = xf_corner + wf * info.dilation_depth;
        if (xf < 0 || xf >= info.in_depth) {
          continue;
        }

        for (int yr = 0; yr < info.out_height; ++yr) {
          int xr_corner = yr * info.stride_height - info.pad_top;
          for (int wr = 0; wr < info.filter_height; ++wr) {
            
          }
        }
      }
    }
  }
}

// template <typename DY, typename DX, typename FM>
// inline void NDHWCPool3DGradImpl(const DY* dy_buf, DX* dx_buf,
//                                 const NDHWCPool3DInfo& info,
//                                 const FM& pixel_mask) {
//   for (int batch = 0; batch < info.batch_size; ++batch) {
//     for (int channel = 0; channel < info.channel_size; ++channel) {
//       for (int dx_depth = 0; dx_depth < info.in_depth; ++dx_depth) {
//         for (int dx_row = 0; dx_row < info.in_height; ++dx_row) {
//           for (int dx_col = 0; dx_col < info.in_width; ++dx_col) {
//             // Sharder code begins
//             int dy_depth_corner = dx_depth - info.pad_front;
//             int dy_row_corner = dx_row - info.pad_top;
//             int dy_col_corner = dx_col - info.pad_left;

//             int dx_offset =
//                 info.in_offset(batch, dx_depth, dx_row, dx_col, channel);
//             DX dot_prod = 0;
//             for (int w_depth = 0; w_depth < info.effective_filter_depth;
//                  w_depth += info.dilation_depth) {
//               int dy_depth = (dy_depth_corner + w_depth) / info.stride_depth;
//               if (int rem = (dy_depth_corner + w_depth) % info.stride_depth;
//                   dy_depth < 0 || dy_depth >= info.out_depth || rem != 0) {
//                 continue;
//               }
//               for (int w_row = 0; w_row < info.effective_filter_height;
//                    w_row += info.dilation_height) {
//                 int dy_row = (dy_row_corner + w_row) / info.stride_height;
//                 if (int rem = (dy_row_corner + w_row) % info.stride_height;
//                     dy_row < 0 || dy_row >= info.out_height || rem != 0) {
//                   continue;
//                 }
//                 for (int w_col = 0; w_col < info.effective_filter_width;
//                      w_col += info.dilation_width) {
//                   int dy_col = (dy_col_corner + w_col) / info.stride_width;
//                   if (int rem = (dy_col_corner + w_col) % info.stride_width;
//                       dy_col < 0 || dy_col >= info.out_width || rem != 0) {
//                     continue;
//                   }

//                   int dy_offset =
//                       info.out_offset(batch, dy_depth, dy_row, dy_col,
//                       channel);
//                   DY pixel = dy_buf[dy_offset];
//                   dot_prod += pixel * pixel_mask(dy_offset, dx_offset);
//                 }
//               }
//             }
//             dx_buf[dx_offset] = dot_prod;
//           }
//         }
//       }
//     }
//   }
// }

}  // namespace tfjs::wasm

#endif
