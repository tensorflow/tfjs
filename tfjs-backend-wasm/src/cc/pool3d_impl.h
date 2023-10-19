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

struct NDHWCPool3DInfo {
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
  int effective_filter_depth;
  int effective_filter_height;
  int effective_filter_width;
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
template <typename IN, typename FI, typename FAP, typename FAG>
inline void NDHWCPool3DImpl(const IN* x_buf, const NDHWCPool3DInfo& info,
                            const FI& filter_init, const FAP& filter_apply,
                            const FAG& filter_assign) {
  for (int batch = 0; batch < info.batch_size; ++batch) {
    for (int channel = 0; channel < info.channel_size; ++channel) {
      for (int y_depth = 0; y_depth < info.out_depth; ++y_depth) {
        int x_depth_corner = y_depth * info.stride_depth - info.pad_front;
        int x_depth_min =
            AddUntilNonNegative(x_depth_corner, info.dilation_depth);
        int x_depth_max = std::min(
            info.in_depth, info.effective_filter_depth + x_depth_corner);

        for (int y_row = 0; y_row < info.out_height; ++y_row) {
          int x_row_corner = y_row * info.stride_height - info.pad_top;
          int x_row_min =
              AddUntilNonNegative(x_row_corner, info.dilation_height);
          int x_row_max = std::min(info.in_height,
                                   info.effective_filter_height + x_row_corner);
          for (int y_col = 0; y_col < info.out_width; ++y_col) {
            int x_col_corner = y_col * info.stride_width - info.pad_left;
            int x_col_min =
                AddUntilNonNegative(x_col_corner, info.dilation_width);
            int x_col_max = std::min(
                info.in_width, info.effective_filter_width + x_col_corner);

            // Apply the filter
            auto filter_data = filter_init();
            for (int x_depth = x_depth_min; x_depth < x_depth_max;
                 x_depth += info.dilation_depth) {
              for (int x_row = x_row_min; x_row < x_row_max;
                   x_row += info.dilation_height) {
                for (int x_col = x_col_min; x_col < x_col_max;
                     x_col += info.dilation_width) {
                  int x_offset =
                      info.in_offset(batch, x_depth, x_row, x_col, channel);
                  filter_apply(filter_data, x_offset, x_buf[x_offset]);
                }
              }
            }
            int out_offset =
                info.out_offset(batch, y_depth, y_row, y_col, channel);
            filter_assign(out_offset, filter_data);
          }
        }
      }
    }
  }
}

template <typename DY, typename DX, typename FM>
inline void NDHWCPool3DGradImpl(const DY* dy_buf, DX* dx_buf,
                                const NDHWCPool3DInfo& forward_info,
                                const FM& pixel_mask) {
  auto info = forward_info;
  info.pad_front = info.effective_filter_depth - 1 - info.pad_front;
  info.pad_top = info.effective_filter_height - 1 - info.pad_top;
  info.pad_left = info.effective_filter_width - 1 - info.pad_left;

  for (int batch = 0; batch < info.batch_size; ++batch) {
    for (int channel = 0; channel < info.channel_size; ++channel) {
      for (int dx_depth = 0; dx_depth < info.in_depth; ++dx_depth) {
        for (int dx_row = 0; dx_row < info.in_height; ++dx_row) {
          for (int dx_col = 0; dx_col < info.in_width; ++dx_col) {
            // Sharder code begins
            int dy_depth_corner = dx_depth - info.pad_front;
            int dy_row_corner = dx_row - info.pad_top;
            int dy_col_corner = dx_col - info.pad_left;

            int dx_offset =
                info.in_offset(batch, dx_depth, dx_row, dx_col, channel);
            DX dot_prod = 0;
            for (int w_depth = 0; w_depth < info.effective_filter_depth;
                 w_depth += info.dilation_depth) {
              int dy_depth = (dy_depth_corner + w_depth) / info.stride_depth;
              if (int rem = (dy_depth_corner + w_depth) % info.stride_depth;
                  dy_depth < 0 || dy_depth >= info.out_depth || rem != 0) {
                continue;
              }
              for (int w_row = 0; w_row < info.effective_filter_height;
                   w_row += info.dilation_height) {
                int dy_row = (dy_row_corner + w_row) / info.stride_height;
                if (int rem = (dy_row_corner + w_row) % info.stride_height;
                    dy_row < 0 || dy_row >= info.out_height || rem != 0) {
                  continue;
                }
                for (int w_col = 0; w_col < info.effective_filter_width;
                     w_col += info.dilation_width) {
                  int dy_col = (dy_col_corner + w_col) / info.stride_width;
                  if (int rem = (dy_col_corner + w_col) % info.stride_width;
                      dy_col < 0 || dy_col >= info.out_width || rem != 0) {
                    continue;
                  }

                  int dy_offset =
                      info.out_offset(batch, dy_depth, dy_row, dy_col, channel);
                  DY pixel = dy_buf[dy_offset];
                  dot_prod += pixel * pixel_mask(dy_offset, dx_offset);
                }
              }
            }
            dx_buf[dx_offset] = dot_prod;
          }
        }
      }
    }
  }
}

}  // namespace tfjs::wasm
