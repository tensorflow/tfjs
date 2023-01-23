#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include <cstddef>

namespace tfjs::wasm {

namespace {

inline int AddUntilNonNegative(int v, int d) {
  if (v >= 0) {
    return v;
  }
  return (v % d + v) % d;
}

}  // namespace

struct Pool2DInfo {
  int batch_size;
  // Since Pool ops (AvgPool and MaxPool) support 2D filter only, in
  // channels should always equal to out channels.
  int channel_size;
  int in_height;
  int in_width;
  int out_height;
  int out_width;

  int stride_height;
  int stride_width;
  int dilation_height;
  int dilation_width;
  int effective_filter_height;
  int effective_filter_width;
  int pad_top;
  int pad_left;

  inline int in_offset(int b, int h, int w, int c) const {
    return c + (w + (h + b * in_height) * in_width) * channel_size;
  }
  inline int out_offset(int b, int h, int w, int c) const {
    return c + (w + (h + b * out_height) * out_width) * channel_size;
  }
  inline int in_size() const {
    return batch_size * in_height * in_width * channel_size;
  }
  inline int out_size() const {
    return batch_size * out_height * out_width * channel_size;
  }
};
template <typename DY, typename DX, typename FM>
inline void Pool2DGradImpl(const DY* dy_buf, DX* dx_buf, const Pool2DInfo& info,
                           const FM& pixel_mask) {
  for (int batch = 0; batch < info.batch_size; ++batch) {
    for (int channel = 0; channel < info.channel_size; ++channel) {
      for (int dx_row = 0; dx_row < info.in_height; ++dx_row) {
        for (int dx_col = 0; dx_col < info.in_width; ++dx_col) {
          // Sharder code begins
          int dy_row_corner =
              dx_row - (info.effective_filter_height - 1 - info.pad_top);
          int dy_col_corner =
              dx_col - (info.effective_filter_width - 1 - info.pad_left);

          int dx_offset = info.in_offset(batch, dx_row, dx_col, channel);
          DX dot_prod = 0;
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

              int dy_offset = info.out_offset(batch, dy_row, dy_col, channel);
              DY pixel = dy_buf[dy_offset];
              dot_prod += pixel * pixel_mask(dy_offset, dx_offset);
            }
          }
          dx_buf[dx_offset] = dot_prod;
        }
      }
    }
  }
}

}  // namespace tfjs::wasm
