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

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include <xnnpack.h>
#include <limits>

#include "src/cc/backend.h"
#include "src/cc/binary.h"
#include "src/cc/util.h"

namespace {
template <class T>
inline T add(T a, T b) {
  return a + b;
}

xnn_operator_t add_op = nullptr;
void xnn_add(const int a_id, const size_t* a_shape_ptr, const int a_shape_len,
             const int b_id, const size_t* b_shape_ptr, const int b_shape_len,
             const int out_id) {
  auto& a_info = tfjs::backend::get_tensor_info(a_id);
  auto& b_info = tfjs::backend::get_tensor_info(b_id);
  auto& out_info = tfjs::backend::get_tensor_info_out(out_id);
  const float* a_buf = a_info.f32();
  const float* b_buf = b_info.f32();
  float* out_buf = out_info.f32_write();
  if (add_op == nullptr) {
    const float sum_min = -std::numeric_limits<float>::infinity(),
                sum_max = std::numeric_limits<float>::infinity();
    const int flags = 0;
    xnn_status status = xnn_create_add_nd_f32(sum_min, sum_max, flags, &add_op);
    if (status != xnn_status_success) {
      tfjs::util::warn(
          "XNN status for xnn_create_add_nd_f32 is not successful. Got "
          "status %d. Use -c dbg to see XNN logs.");
      return;
    }
    tfjs::backend::xnn_operator_count++;
  }
  const int batch_size = out_info.size;
  xnn_status status = xnn_setup_add_nd_f32(
      add_op, a_shape_len, a_shape_ptr, b_shape_len, b_shape_ptr, a_buf, b_buf,
      out_buf, nullptr /* thread pool */);
  if (status != xnn_status_success) {
    tfjs::util::warn(
        "XNN status for xnn_setup_add_nd_f32 is not successful. Got "
        "status %d. Use -c dbg to see XNN logs.",
        status);
    return;
  }

  xnn_run_operator(add_op, nullptr /* thread pool */);
}
}  // namespace

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void Add(const int a_id, const size_t* a_shape_ptr, const int a_shape_len,
         const int b_id, const size_t* b_shape_ptr, const int b_shape_len,
         const DType dtype, const int out_id) {
  switch (dtype) {
    case DType::float32:
      xnn_add(a_id, a_shape_ptr, a_shape_len, b_id, b_shape_ptr, b_shape_len,
              out_id);
      break;
    case DType::int32:
      binary_i32(a_id, b_id, out_id, add<int>);
      break;
    case DType::boolean:
      binary_bool(a_id, b_id, out_id, add<bool>);
      break;
    default:
      util::warn("Add for tensor ids %d and %d failed. Unknown dtype %d", a_id,
                 b_id, dtype);
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
