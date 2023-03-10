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

#ifndef CONV3D_IMPL_H_
#define CONV3D_IMPL_H_

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include <cstddef>
#include "tfjs-backend-wasm/src/cc/shape.h"

namespace tfjs::wasm {

struct NDHWCConv3DInfo {
  int batch_size;
  int in_depth;
  int in_height;
  int in_width;
  int in_channels;
  int out_depth;
  int out_height;
  int out_width;
  int out_channels;

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
        {batch_size, in_depth, in_height, in_width, in_channels});
  }
  inline Shape<int, 5> out_shape() const {
    return Shape<int, 5>(
        {batch_size, out_depth, out_height, out_width, out_channels});
  }
  inline Shape<int, 5> filter_shape() const {
    return Shape<int, 5>(
        {filter_depth, filter_height, filter_width, in_channels, out_channels});
  }

  inline int in_offset(int b, int d, int h, int w, int c) const {
    return in_shape().offset({b, d, h, w, c});
  }
  inline int out_offset(int b, int d, int h, int w, int c) const {
    return out_shape().offset({b, d, h, w, c});
  }
  inline int filter_offset(int d, int h, int w, int c1, int c2) const {
    return filter_shape().offset({d, h, w, c1, c2});
  }

  inline int int_size() const { return in_shape().size(); }
  inline int out_size() const { return out_shape().size(); }
};

template <typename IN, typename FILTER, typename OUT>
inline void NDHWCConv3DImpl(const IN* x_buf, const FILTER* filter_buf,
                            OUT* out_buf, const NDHWCConv3DInfo& info) {
  memset(out_buf, 0, sizeof(OUT) * info.out_size());
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
            int xr = xr_corner + wr * info.dilation_height;
            if (xr < 0 || xr >= info.in_height) {
              continue;
            }

            for (int yc = 0; yc < info.out_width; ++yc) {
              int xc_corner = yc * info.stride_width - info.pad_left;
              for (int wc = 0; wc < info.filter_width; ++wc) {
                int xc = xc_corner + wc * info.dilation_width;
                if (xc < 0 || xc >= info.in_width) {
                  continue;
                }

                for (int d1 = 0; d1 < info.in_channels; ++d1) {
                  for (int d2 = 0; d2 < info.out_channels; ++d2) {
                    out_buf[info.out_offset(b, yf, yr, yc, d2)] +=
                        x_buf[info.in_offset(b, xf, xr, xc, d1)] *
                        filter_buf[info.filter_offset(wf, wr, wc, d1, d2)];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

namespace {

inline int DivCeil(int a, int b) {
  return (a / b) + static_cast<int>(a % b != 0);
}
inline int DivFloor(int a, int b) { return a / b; }

}  // namespace

template <typename IN, typename FILTER, typename OUT>
inline void NDHWCConv3DBackpropFilterV2Impl(
    const IN* x_buf, const OUT* dy_buf, FILTER* dw_buf,
    const NDHWCConv3DInfo& forward_info) {
  const NDHWCConv3DInfo& info = forward_info;
  for (int wf = 0; wf < info.filter_depth; ++wf) {
    int yf_min = std::max(0, DivCeil(info.pad_front - wf, info.stride_depth));
    int yf_max = std::min(
        info.out_depth,
        DivFloor(info.in_depth + info.pad_front - wf, info.stride_depth));

    for (int wr = 0; wr < info.filter_height; ++wr) {
      int yr_min = std::max(0, DivCeil(info.pad_top - wr, info.stride_height));
      int yr_max = std::min(
          info.out_height,
          DivFloor(info.in_height + info.pad_top - wr, info.stride_height));

      for (int wc = 0; wc < info.filter_width; ++wc) {
        int yc_min =
            std::max(0, DivCeil(info.pad_left - wc, info.stride_width));
        int yc_max = std::min(
            info.out_width,
            DivFloor(info.in_width + info.pad_left - wc, info.stride_width));

        for (int d1 = 0; d1 < info.in_channels; ++d1) {
          for (int d2 = 0; d2 < info.out_channels; ++d2) {
            FILTER dot_prod = 0;
            for (int b = 0; b < info.batch_size; ++b) {
              for (int yf = yf_min; yf < yf_max; ++yf) {
                for (int yr = yr_min; yr < yr_max; ++yr) {
                  for (int yc = yc_min; yc < yc_max; ++yc) {
                    int xf = wf + yf * info.stride_depth - info.pad_front;
                    int xr = wr + yr * info.stride_height - info.pad_top;
                    int xc = wc + yc * info.stride_width - info.pad_left;
                    dot_prod += x_buf[info.in_offset(b, xf, xr, xc, d1)] *
                                dy_buf[info.out_offset(b, yf, yr, yc, d2)];
                  }
                }
              }
            }
            dw_buf[info.filter_offset(wf, wr, wc, d1, d2)] = dot_prod;
          }
        }
      }
    }
  }
}

template <typename IN, typename FILTER, typename OUT>
inline void NDHWCConv3DBackpropInputV2Impl(
    const FILTER* filter_buf, const OUT* dy_buf, IN* dx_buf,
    const NDHWCConv3DInfo& forward_info) {
  NDHWCConv3DInfo info = forward_info;
  info.pad_front = info.filter_depth - 1 - info.pad_front;
  info.pad_top = info.filter_height - 1 - info.pad_top;
  info.pad_left = info.filter_width - 1 - info.pad_left;

  for (int b = 0; b < info.batch_size; ++b) {
    for (int d1 = 0; d1 < info.in_channels; ++d1) {
      // Frames of depth
      for (int xf = 0; xf < info.in_depth; ++xf) {
        int xf_corner = xf - info.pad_front;
        int xf_min = std::max(0, DivCeil(xf_corner, info.stride_depth));
        int yf_max = std::min(info.out_depth,
                              DivFloor(info.filter_depth, info.stride_depth));

        // Rows as per standard 2d matrix notation
        for (int xr = 0; xr < info.in_height; ++xr) {
          int xr_corner = xr - info.pad_top;
          int xr_min = std::max(0, DivCeil(xr_corner, info.stride_height));
          int yr_max = std::min(
              info.out_height,
              DivFloor(info.filter_height + xr_corner, info.stride_height));

          // Columns as per standard 2d matrix notation
          for (int xc = 0; xc < info.in_width; ++xc) {
            int xc_corner = xc - info.pad_left;
            int xc_min = std::max(0, DivCeil(xc_corner, info.stride_width));
            int yc_max = std::min(
                info.out_width,
                DivFloor(info.filter_width + xc_corner, info.stride_width));

            IN dot_prod = 0;
            for (int yf = xf_min; yf < yf_max; ++yf) {
              for (int yr = xr_min; yr < yr_max; ++yr) {
                for (int yc = xc_min; yc < yc_max; ++yc) {
                  int wf = yf * info.stride_depth - xf_corner;
                  int wr = yr * info.stride_height - xr_corner;
                  int wc = yc * info.stride_width - xc_corner;

                  for (int d2 = 0; d2 < info.out_channels; ++d2) {
                    dot_prod += dy_buf[info.out_offset(b, yf, yr, yc, d2)] *
                                filter_buf[info.filter_offset(
                                    info.filter_depth - 1 - wf,
                                    info.filter_height - 1 - wr,
                                    info.filter_width - 1 - wc, d1, d2)];
                  }
                }
              }
            }
            dx_buf[info.in_offset(b, xf, xr, xc, d1)] = dot_prod;
          }
        }
      }
    }
  }
}

}  // namespace tfjs::wasm

#endif
