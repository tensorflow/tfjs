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
#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/bincount_impl.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace tfjs::wasm {

namespace {

template <typename T>
inline void Bincount2D(const int32_t* x_buf, int32_t x_shape_0,
                       int32_t x_shape_1, int32_t size, const T* weight_buf,
                       bool binary_output, T* out_buf) {
  std::fill(out_buf, out_buf + (x_shape_0 * size), 0);
  for (int32_t i = 0; i < x_shape_0; ++i) {
    BincountImpl</*reset_out_buf=*/false>(
        x_buf + i * x_shape_1, x_shape_1, size,
        weight_buf != nullptr ? weight_buf + i * x_shape_1 : nullptr,
        binary_output, out_buf + i * size);
  }
}

}  // namespace

extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

// REQUIRES:
// - Tensor `x` must have dtype int32 (checked in tfjs-core)
// - Tensor `x` must be 1D or 2D (checked in tfjs-core)
// - If has_weights is true, tensor `weights` must have the same shape as `x`
// (checked in tfjs-core)
// - Tensor `out` must have shape [x.shape[0], size] or [size]
// - Tensor `out` must have the same dtype as weights
void DenseBincount(const int32_t x_id, const int32_t* x_shape_ptr,
                   const int32_t x_shape_len, const int32_t size,
                   const bool has_weights, const int32_t weights_id,
                   const DType weights_dtype, const bool binary_output,
                   const int32_t out_id) {
  const TensorInfo& x_info = backend::get_tensor_info(x_id);
  const TensorInfo* weights_info =
      has_weights ? &backend::get_tensor_info(weights_id) : nullptr;
  TensorInfo& out_info = backend::get_tensor_info_out(out_id);

  const int32_t* x_buf = x_info.i32();
  if (x_shape_len == 1) {
    switch (weights_dtype) {
      case DType::float32:
        BincountImpl(x_buf, x_shape_ptr[0], size,
                     weights_info ? weights_info->f32() : nullptr,
                     binary_output, out_info.f32_write());
        break;
      case DType::int32:
        BincountImpl(x_buf, x_shape_ptr[0], size,
                     weights_info ? weights_info->i32() : nullptr,
                     binary_output, out_info.i32_write());
        break;
      default:
        util::warn(
            "DenseBincount with weights tensor id %d failed. Unsupported "
            "weights "
            "dtype %d",
            weights_id, weights_dtype);
    }
    return;
  }

  // x_shape_len == 2
  switch (weights_dtype) {
    case DType::float32:
      Bincount2D(x_buf, x_shape_ptr[0], x_shape_ptr[1], size,
                 weights_info ? weights_info->f32() : nullptr, binary_output,
                 out_info.f32_write());
      break;
    case DType::int32:
      Bincount2D(x_buf, x_shape_ptr[0], x_shape_ptr[1], size,
                 weights_info ? weights_info->i32() : nullptr, binary_output,
                 out_info.i32_write());
      break;
    default:
      util::warn(
          "DenseBincount with weights tensor id %d failed. Unsupported "
          "weights "
          "dtype %d",
          weights_id, weights_dtype);
  }
}

}  // extern "C"
}  // namespace tfjs::wasm
