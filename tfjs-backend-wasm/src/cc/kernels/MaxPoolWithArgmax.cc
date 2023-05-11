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
#include <limits>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/pool3d_impl.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace tfjs::wasm {

namespace {

template <typename T>
inline void MaxPoolWithArgmaxImpl(const T* x_buf, T* pooled_buf,
                                  int32_t* indexes_buf,
                                  bool include_batch_index,
                                  const NDHWCPool3DInfo& pool_info) {
  NDHWCPool3DImpl(x_buf, pool_info,
                  /*filter_init=*/
                  []() -> std::pair<T, int> {
                    return {std::numeric_limits<T>::min(), 0};
                  },
                  /*filter_apply=*/
                  [](std::pair<T, int>& data, int x_offset, const T& x_val) {
                    if (x_val >= data.first) {
                      data = {x_val, x_offset};
                    }
                  },
                  /*filter_assign=*/
                  [pooled_buf, indexes_buf, include_batch_index,
                   index_mod = pool_info.in_height * pool_info.in_width *
                               pool_info.channel_size](
                      int offset, const std::pair<T, int>& data) {
                    pooled_buf[offset] = data.first;
                    indexes_buf[offset] = include_batch_index
                                              ? data.second
                                              : data.second % index_mod;
                  });
}

}  // namespace

// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

// REQUIRES:
// - Tensor `x` and `out` must have the same dtype.
// - Tensor `indexes` must have dtype `int32`.
void MaxPoolWithArgmax(int x_id, int pooled_id, int indexes_id, DType dtype,
                       bool include_batch_index, int batch_size,
                       int channel_size, int in_height, int in_width,
                       int out_height, int out_width, int stride_height,
                       int stride_width, int dilation_height,
                       int dilation_width, int effective_filter_height,
                       int effective_filter_width, int pad_top, int pad_left) {
  const TensorInfo& x_info = backend::get_tensor_info(x_id);
  TensorInfo& pooled_info = backend::get_tensor_info_out(pooled_id);
  TensorInfo& indexes_info = backend::get_tensor_info_out(indexes_id);

  NDHWCPool3DInfo pool_info{
      .batch_size = batch_size,
      .channel_size = channel_size,
      .in_depth = 1,
      .in_height = in_height,
      .in_width = in_width,
      .out_depth = 1,
      .out_height = out_height,
      .out_width = out_width,
      .stride_depth = 1,
      .stride_height = stride_height,
      .stride_width = stride_width,
      .dilation_depth = 1,
      .dilation_height = dilation_height,
      .dilation_width = dilation_width,
      .effective_filter_depth = 1,
      .effective_filter_height = effective_filter_height,
      .effective_filter_width = effective_filter_width,
      .pad_front = 0,
      .pad_top = pad_top,
      .pad_left = pad_left,
  };

  switch (dtype) {
    case DType::float32:
      MaxPoolWithArgmaxImpl(x_info.f32(), pooled_info.f32_write(),
                            indexes_info.i32_write(), include_batch_index,
                            pool_info);
      break;
    case DType::int32:
      MaxPoolWithArgmaxImpl(x_info.i32(), pooled_info.i32_write(),
                            indexes_info.i32_write(), include_batch_index,
                            pool_info);
      break;
    case DType::boolean:
      MaxPoolWithArgmaxImpl(x_info.b(), pooled_info.b_write(),
                            indexes_info.i32_write(), include_batch_index,
                            pool_info);
      break;
    default:
      util::warn("MaxPoolWithArgmax for tensor id failed. Unknown dtype %d",
                 x_id, dtype);
  }
}

}  // extern "C"
}  // namespace tfjs::wasm
