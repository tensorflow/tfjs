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

#include <algorithm>
#include <cstddef>
#include <limits>
#include <tuple>

namespace tfjs::wasm {

struct Dilation2DInfo {
  int batch;
  int depth;
  int in_height;
  int in_width;
  int out_height;
  int out_width;
  int stride_height;
  int stride_width;
  int dilation_height;
  int dilation_width;
  int filter_height;
  int filter_width;
  int pad_top;
  int pad_left;

  inline int in_offset(int b, int h, int w, int d) const {
    return d + (w + (h + b * in_height) * in_width) * depth;
  }
  inline int filter_offset(int h, int w, int d) const { return d + (w + h * filter_width) * depth; }
  inline int out_offset(int b, int h, int w, int d) const {
    return d + (w + (h + b * out_height) * out_width) * depth;
  }

  inline int in_size() const { return batch * in_height * in_width * depth; }
  inline int filter_size() const { return filter_height * filter_width * depth; }
  inline int out_size() const { return batch * out_height * out_width * depth; }
};

namespace {

// Internal implementation for iterating through loops in Dilation2D ops.
template <typename FI, typename FAP, typename FAG>
inline void IterateDilation2DLoops(const Dilation2DInfo& info, const FI& filter_init,
                                   const FAP& filter_apply, const FAG& filter_aggregate) {
  for (int b = 0; b < info.batch; ++b) {
    for (int h_out = 0; h_out < info.out_height; ++h_out) {
      int h_beg = h_out * info.stride_height - info.pad_top;
      for (int w_out = 0; w_out < info.out_width; ++w_out) {
        int w_beg = w_out * info.stride_width - info.pad_left;
        for (int d = 0; d < info.depth; ++d) {
          auto data = filter_init(h_beg, w_beg);

          for (int h = 0; h < info.filter_height; ++h) {
            int h_in = h_beg + h * info.dilation_height;
            if (h_in < 0 || h_in >= info.in_height) {
              continue;
            }

            for (int w = 0; w < info.filter_width; ++w) {
              int w_in = w_beg + w * info.dilation_width;
              if (w_in < 0 || w_in >= info.in_width) {
                continue;
              }
              filter_apply(data, b, d, h_in, w_in, h, w);
            }
          }

          filter_aggregate(data, b, d, h_out, w_out);
        }
      }
    }
  }
}

}  // namespace

template <typename T>
inline void Dilation2DImpl(const T* x_buf, const T* filter_buf, T* out_buf,
                           const Dilation2DInfo& info) {
  IterateDilation2DLoops(
      info,
      /*filter_init=*/
      [](int /*h_beg*/, int /*w_beg*/) -> T { return std::numeric_limits<T>::min(); },
      /*filter_apply=*/
      [&](T& cur_val, int b, int d, int h_in, int w_in, int h, int w) {
        int x_offset = info.in_offset(b, h_in, w_in, d);
        int filter_offset = info.filter_offset(h, w, d);
        cur_val = std::max(x_buf[x_offset] + filter_buf[filter_offset], cur_val);
      },
      /*filter_aggregate=*/
      [&](const T& cur_val, int b, int d, int h_out, int w_out) {
        int out_offset = info.out_offset(b, h_out, w_out, d);
        out_buf[out_offset] = cur_val;
      });
}

template <typename T>
inline void Dilation2DBackpropFilterImpl(const T* x_buf, const T* filter_buf, const T* dy_buf,
                                         T* grad_buf, const Dilation2DInfo& info) {
  std::fill(grad_buf, grad_buf + info.filter_size(), 0);
  IterateDilation2DLoops(
      info,
      /*filter_init=*/
      [](int /*h_beg*/, int /*w_beg*/) -> std::tuple<T, int, int> {
        return {std::numeric_limits<T>::min(), 0, 0};
      },
      /*filter_apply=*/
      [&](std::tuple<T, int, int>& data, int b, int d, int h_in, int w_in, int h, int w) {
        int x_offset = info.in_offset(b, h_in, w_in, d);
        int filter_offset = info.filter_offset(h, w, d);
        T val = x_buf[x_offset] + filter_buf[filter_offset];
        if (val > std::get<0>(data)) {
          data = {val, h, w};
        }
      },
      /*filter_aggregate=*/
      [&](const std::tuple<T, int, int>& data, int b, int d, int h_out, int w_out) {
        const auto& [unused, h_max, w_max] = data;
        const int dy_offset = info.out_offset(b, h_out, w_out, d);
        int grad_offset = info.filter_offset(h_max, w_max, d);
        grad_buf[grad_offset] += dy_buf[dy_offset];
      });
}

template <typename T>
inline void Dilation2DBackpropInputImpl(const T* x_buf, const T* filter_buf, const T* dy_buf,
                                        T* grad_buf, const Dilation2DInfo& info) {
  std::fill(grad_buf, grad_buf + info.in_size(), 0);
  IterateDilation2DLoops(
      info,
      /*filter_init=*/
      [](int h_beg, int w_beg) -> std::tuple<T, int, int> {
        return {std::numeric_limits<T>::min(), std::max(h_beg, 0), std::max(w_beg, 0)};
      },
      /*filter_apply=*/
      [&](std::tuple<T, int, int>& data, int b, int d, int h_in, int w_in, int h, int w) {
        int x_offset = info.in_offset(b, h_in, w_in, d);
        int filter_offset = info.filter_offset(h, w, d);
        T val = x_buf[x_offset] + filter_buf[filter_offset];
        if (val > std::get<0>(data)) {
          data = {val, h_in, w_in};
        }
      },
      /*filter_aggregate=*/
      [&](const std::tuple<T, int, int>& data, int b, int d, int h_out, int w_out) {
        const auto& [unused, h_in_max, w_in_max] = data;
        const int dy_offset = info.out_offset(b, h_out, w_out, d);
        int grad_offset = info.in_offset(b, h_in_max, w_in_max, d);
        grad_buf[grad_offset] += dy_buf[dy_offset];
      });
}

}  // namespace tfjs::wasm
