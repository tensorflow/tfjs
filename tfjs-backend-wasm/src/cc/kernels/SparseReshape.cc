/* Copyright 2021 Google LLC. All Rights Reserved.
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

#include <cmath>
#include <cstddef>
#include <vector>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace tfjs {
namespace wasm {
extern "C" {
#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void SparseReshape(const size_t input_indices_id, const size_t input_shape_id,
                   const size_t target_shape_id, const size_t nnz,
                   const size_t new_indices_id, const size_t output_shape_id,
                   const size_t exception_values_id) {
  auto& input_indices_info = backend::get_tensor_info(input_indices_id);
  auto& input_shape_info = backend::get_tensor_info(input_shape_id);
  auto& target_shape_info = backend::get_tensor_info_out(target_shape_id);
  auto& new_indices_info = backend::get_tensor_info_out(new_indices_id);
  auto& output_shape_info = backend::get_tensor_info_out(output_shape_id);
  auto& exception_values_info =
      backend::get_tensor_info_out(exception_values_id);

  const int32_t* input_indices = input_indices_info.i32();
  const int32_t* input_shape = input_shape_info.i32();
  const int32_t* target_shape = target_shape_info.i32();
  int32_t* new_indices = new_indices_info.i32_write();
  int32_t* output_shape = output_shape_info.i32_write();
  int32_t* exception_values = exception_values_info.i32_write();

  // Initialize to no exceptions.
  exception_values[0] = -1;

  const int32_t dense_size =
      util::size_from_shape({input_shape, input_shape + input_shape_info.size});
  const int32_t output_rank = target_shape_info.size;

  // Compute the output shape. Determine product of specified dimensions, and
  // find the index of the unspecified one.
  int32_t product = 1;
  int32_t unknown_index = -1;
  for (int32_t d = 0; d < output_rank; ++d) {
    const int32_t size = target_shape[d];
    if (size == -1) {
      if (unknown_index != -1) {
        exception_values[0] = 0;
        exception_values[1] = unknown_index;
        exception_values[2] = d;
        return;
      }
      unknown_index = d;
      output_shape[d] = 1;
    } else {
      if (size < 0) {
        exception_values[0] = 1;
        exception_values[1] = d;
        exception_values[2] = size;
        return;
      }
      product *= size;
      output_shape[d] = size;
    }
  }
  if (unknown_index != -1) {
    if (product <= 0) {
      exception_values[0] = 2;
      return;
    }
    const int32_t missing =
        std::trunc(static_cast<double>(dense_size) / product);
    if (product * missing != dense_size) {
      exception_values[0] = 3;
      exception_values[1] = dense_size;
      exception_values[2] = product;
      return;
    }

    output_shape[unknown_index] = missing;
  }
  const int32_t output_size =
      util::size_from_shape({output_shape, output_shape + output_rank});
  if (output_size != dense_size) {
    exception_values[0] = 4;
    exception_values[1] = dense_size;
    exception_values[2] = output_size;
    return;
  }

  const int32_t input_rank = input_shape_info.size;
  std::vector<int32_t> input_strides(input_rank);
  if (input_rank > 0) {
    input_strides[input_rank - 1] = 1;
    for (int32_t d = input_rank - 2; d >= 0; --d) {
      input_strides[d] = input_strides[d + 1] * input_shape[d + 1];
    }
  }

  std::vector<int32_t> output_strides(output_rank);
  if (output_rank > 0) {
    output_strides[output_rank - 1] = 1;
    for (int32_t d = output_rank - 2; d >= 0; --d) {
      output_strides[d] = output_strides[d + 1] * output_shape[d + 1];
    }
  }

  for (size_t i = 0; i < nnz; ++i) {
    size_t id = 0;
    for (size_t j = 0; j < input_rank; ++j) {
      // input_indices is a 2d tensor with shape of [nnz, input_rank]
      id += input_indices[i * input_rank + j] * input_strides[j];
    }
    for (size_t j = 0; j < output_rank; ++j) {
      // new_indices is a 2d tensor with shape of [nnz, output_rank]
      new_indices[i * output_rank + j] =
          std::trunc(static_cast<double>(id) / output_strides[j]);
      id %= output_strides[j];
    }
  }
}
}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
