#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include <cmath>
#include <cstddef>
#include <functional>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace tfjs::wasm {

namespace {

inline int AddUntilNonNegative(int v, int d) {
  if (v >= 0) {
    return v;
  }
  return (v % d) + v;
}

}  // namespace

struct Pool3DInfo {
  int stride_batch;
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
  int in_channels;
  int in_depth;
  int in_height;
  int out_depth;
};
template <typename OUT, typename IN, typename T>
inline void Pool3DImpl(
    const Pool3DInfo& info, const std::function<T(void)>& filter_init,
    const std::function<void(const T&, const IN&)>& filter_apply,
    const std::function<OUT(const T&)>& filter_aggregate) {
  for (int batch = 0; batch < info.batch_size; ++batch) {
    for (int channel = 0; channel < info.in_channels; ++channel) {
      for (int y_depth = 0; y_depth < info.out_depth; ++y_depth) {
        int x_depth_corner = y_depth * info.stride_depth - info.pad_front;
        int x_depth_min =
            AddUntilNonNegative(x_depth_corner, info.dilation_depth);
        int x_depth_max = std::min(
            info.in_depth, info.effective_filter_depth + info.x_depth_corner);

        for (int y_row = 0; y_row < info.out_height; ++y_row) {
          int x_row_corner = y_row * info.stride_height - info.pad_top;
          int x_row_min =
              AddUntilNonNegative(x_row_corner, info.dilation_height);
          int x_row_max = std::min(info.in_height,
                                   info.effective_filter_height + x_row_corner);
          for (int y_col = 0; y_col < info.out_width; ++y_col) {
            int x_col_corner = y_col * info.info.stride_width - info.pad_left;
            int x_col_min =
                AddUntilNonNegative(x_col_corner, info.dilation_width);
            int x_col_max = std::min(
                info.in_width, info.effective_filter_width + x_col_corner);

            // Apply the filter
            T filter_data = filter_init();
            for (int x_depth = x_depth_min; x_depth < x_depth_max;
                 x_depth += info.dilation_depth) {
              for (int x_row = x_row_min; x_row < x_row_max;
                   x_row += info.dilation_height) {
                for (int x_col = x_col_min; x_col < x_col_max;
                     x_col += info.dilation_width) {
                  int x_offset =
                      (((batch * info.in_depth + x_depth) * info.in_height +
                        x_row) *
                           info.in_width +
                       x_col) *
                          info.in_channels +
                      channel;
                  filter_apply(filter_data, x_buf[x_offset]);
                }
              }
            }
            int out_offset =

            +channel;
            out_buf[out_offset] = filter_aggregate(filter_data);
          }
        }
      }
    }
  }
}

}  // namespace tfjs::wasm
