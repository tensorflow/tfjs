/* Copyright 2019 Google LLC. All Rights Reserved.
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

#include "tfjs-backend-wasm/src/cc/binary.h"

#include <xnnpack.h>
#include <cstddef>
#include <limits>
#include <unordered_map>
#include <vector>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace {
// Maps an `xnn_create_*_nd_f32` function pointer to an instantiated operator.
std::unordered_map<tfjs::wasm::xnn_create_binary_op, xnn_operator_t> op_cache;
}  // namespace

namespace tfjs {
namespace wasm {

template <class I, class O>
void binary_impl(const I* a_buf, const size_t a_size, const I* b_buf,
                 const size_t b_size, O* out_buf, O operation(I, I),
                 const size_t* a_shape_ptr, const size_t a_rank,
                 const size_t* b_shape_ptr, const size_t b_rank) {
  std::vector<size_t> a_shape(a_shape_ptr, a_shape_ptr + a_rank);
  std::vector<size_t> b_shape(b_shape_ptr, b_shape_ptr + b_rank);
  const std::vector<size_t> new_shape =
      tfjs::util::assert_and_get_broadcast_shape(a_shape, b_shape);
  const std::vector<size_t> result_strides =
      tfjs::util::compute_strides(new_shape);
  const size_t result_size = tfjs::util::size_from_shape(new_shape);
  const std::vector<size_t> a_strides = tfjs::util::compute_strides(a_shape);
  const std::vector<size_t> b_strides = tfjs::util::compute_strides(b_shape);
  const std::vector<size_t> a_broadcast_dims =
      tfjs::util::get_broadcast_dims(a_shape, new_shape);
  const std::vector<size_t> b_broadcast_dims =
      tfjs::util::get_broadcast_dims(b_shape, new_shape);

  if (a_broadcast_dims.size() == 0 && b_broadcast_dims.size() == 0) {
    for (size_t i = 0; i < result_size; ++i) {
      out_buf[i] = operation(a_buf[i % a_size], b_buf[i % b_size]);
    }
  } else {
    for (size_t i = 0; i < result_size; ++i) {
      const std::vector<size_t> loc =
          tfjs::util::offset_to_loc(i, result_strides);

      std::vector<size_t> a_loc =
          std::vector<size_t>(loc.end() - a_rank, loc.end());
      for (size_t j = 0; j < a_broadcast_dims.size(); ++j) {
        const size_t d = a_broadcast_dims[j];
        a_loc[d] = 0;
      }
      const size_t a_idx = tfjs::util::loc_to_offset(a_loc, a_strides);

      std::vector<size_t> b_loc =
          std::vector<size_t>(loc.end() - b_rank, loc.end());
      for (size_t k = 0; k < b_broadcast_dims.size(); ++k) {
        const size_t d = b_broadcast_dims[k];
        b_loc[d] = 0;
      }
      const size_t b_idx = tfjs::util::loc_to_offset(b_loc, b_strides);
      out_buf[i] = operation(a_buf[a_idx], b_buf[b_idx]);
    }
  }
}

void binary_f32(const int a_id, const size_t* a_shape_ptr,
                const size_t a_shape_len, const int b_id,
                const size_t* b_shape_ptr, const size_t b_shape_len,
                const size_t out_id, float operation(float, float)) {
  auto& a_info = backend::get_tensor_info(a_id);
  auto& b_info = backend::get_tensor_info(b_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  binary_impl<float, float>(a_info.f32(), a_info.size, b_info.f32(),
                            b_info.size, out_info.f32_write(), operation,
                            a_shape_ptr, a_shape_len, b_shape_ptr, b_shape_len);
}

void binary_i32(const int a_id, const size_t* a_shape_ptr,
                const size_t a_shape_len, const int b_id,
                const size_t* b_shape_ptr, const size_t b_shape_len,
                const size_t out_id, int operation(int, int)) {
  auto& a_info = backend::get_tensor_info(a_id);
  auto& b_info = backend::get_tensor_info(b_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  binary_impl<int32_t, int32_t>(a_info.i32(), a_info.size, b_info.i32(),
                                b_info.size, out_info.i32_write(), operation,
                                a_shape_ptr, a_shape_len, b_shape_ptr,
                                b_shape_len);
}

void binary_bool(const int a_id, const size_t* a_shape_ptr,
                 const size_t a_shape_len, const int b_id,
                 const size_t* b_shape_ptr, const size_t b_shape_len,
                 const size_t out_id, bool operation(bool, bool)) {
  auto& a_info = backend::get_tensor_info(a_id);
  auto& b_info = backend::get_tensor_info(b_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  binary_impl<bool, bool>(a_info.b(), a_info.size, b_info.b(), b_info.size,
                          out_info.b_write(), operation, a_shape_ptr,
                          a_shape_len, b_shape_ptr, b_shape_len);
}

void compare_f32(const int a_id, const size_t* a_shape_ptr,
                 const size_t a_shape_len, const int b_id,
                 const size_t* b_shape_ptr, const size_t b_shape_len,
                 const int out_id, bool operation(float, float)) {
  auto& a_info = backend::get_tensor_info(a_id);
  auto& b_info = backend::get_tensor_info(b_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  binary_impl<float, bool>(a_info.f32(), a_info.size, b_info.f32(), b_info.size,
                           out_info.b_write(), operation, a_shape_ptr,
                           a_shape_len, b_shape_ptr, b_shape_len);
}

void compare_i32(const int a_id, const size_t* a_shape_ptr,
                 const size_t a_shape_len, const int b_id,
                 const size_t* b_shape_ptr, const size_t b_shape_len,
                 const int out_id, bool operation(int, int)) {
  auto& a_info = backend::get_tensor_info(a_id);
  auto& b_info = backend::get_tensor_info(b_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  binary_impl<int, bool>(a_info.i32(), a_info.size, b_info.i32(), b_info.size,
                         out_info.b_write(), operation, a_shape_ptr,
                         a_shape_len, b_shape_ptr, b_shape_len);
}

void compare_bool(const int a_id, const size_t* a_shape_ptr,
                  const size_t a_shape_len, const int b_id,
                  const size_t* b_shape_ptr, const size_t b_shape_len,
                  const int out_id, bool operation(bool, bool)) {
  auto& a_info = backend::get_tensor_info(a_id);
  auto& b_info = backend::get_tensor_info(b_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  binary_impl<bool, bool>(a_info.b(), a_info.size, b_info.b(), b_info.size,
                          out_info.b_write(), operation, a_shape_ptr,
                          a_shape_len, b_shape_ptr, b_shape_len);
}

void logical(const int a_id, const size_t* a_shape_ptr,
             const size_t a_shape_len, const int b_id,
             const size_t* b_shape_ptr, const size_t b_shape_len,
             const int out_id, bool operation(bool, bool)) {
  auto& a_info = backend::get_tensor_info(a_id);
  auto& b_info = backend::get_tensor_info(b_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  binary_impl<bool, bool>(a_info.b(), a_info.size, b_info.b(), b_info.size,
                          out_info.b_write(), operation, a_shape_ptr,
                          a_shape_len, b_shape_ptr, b_shape_len);
}

void binary_xnn_f32(const size_t a_id, const size_t* a_shape_ptr,
                    const size_t a_shape_len, const size_t b_id,
                    const size_t* b_shape_ptr, const size_t b_shape_len,
                    const size_t out_id, xnn_create_binary_op create_op,
                    xnn_setup_binary_op setup_op) {
  auto& a_info = backend::get_tensor_info(a_id);
  auto& b_info = backend::get_tensor_info(b_id);
  auto& out_info = backend::get_tensor_info_out(out_id);
  const float* a_buf = a_info.f32();
  const float* b_buf = b_info.f32();
  float* out_buf = out_info.f32_write();

  xnn_operator_t binary_op = nullptr;

  auto cache_result = op_cache.find(create_op);
  if (cache_result == op_cache.end()) {
    const float sum_min = -std::numeric_limits<float>::infinity(),
                sum_max = std::numeric_limits<float>::infinity();
    const uint32_t flags = 0;
    xnn_status status = create_op(sum_min, sum_max, flags, &binary_op);
    if (status != xnn_status_success) {
      util::warn(
          "XNN status for xnn_create_*_nd_f32 is not successful. Got "
          "status %d. Use -c dbg to see XNN logs.");
      return;
    }
    op_cache.insert({create_op, binary_op});
    backend::xnn_operator_count++;
  } else {
    binary_op = cache_result->second;
  }
  xnn_status status =
      setup_op(binary_op, a_shape_len, a_shape_ptr, b_shape_len, b_shape_ptr,
               a_buf, b_buf, out_buf, tfjs::backend::threadpool);
  if (status != xnn_status_success) {
    util::warn(
        "XNN status for xnn_setup_*_nd_f32 is not successful. Got "
        "status %d. Use -c dbg to see XNN logs.",
        status);
    return;
  }

  xnn_run_operator(binary_op, tfjs::backend::threadpool);
}

}  // namespace wasm
}  // namespace tfjs
