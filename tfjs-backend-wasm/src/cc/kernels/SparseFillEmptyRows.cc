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
#include <numeric>
#include <vector>

#include "tfjs-backend-wasm/src/cc/backend.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace {

template <typename T>
size_t sparse_fill_empty_rows(const int32_t* indices,
                              const size_t indices_count, const size_t rank,
                              const T* values, const size_t dense_rows,
                              const T default_value, int32_t* output_indices,
                              T* output_values, bool* empty_row_indicator,
                              int32_t* reverse_index_map,
                              int32_t* exception_values) {
  // Initialize to no exceptions.
  exception_values[0] = 0;

  if (dense_rows == 0) {
    if (indices_count != 0) {
      exception_values[0] = 1;
      exception_values[1] = indices_count;
    }
    return 0;
  }

  bool rows_are_ordered = true;
  size_t last_indices_row = 0;
  std::vector<size_t> csr_offset(dense_rows, 0);

  for (size_t i = 0; i < indices_count; ++i) {
    // indices is a 2d tensor with shape of [N, rank]
    const int32_t row = indices[i * rank];
    if (row < 0) {
      exception_values[0] = 2;
      exception_values[1] = i;
      exception_values[2] = row;
      return 0;
    }
    if (row >= dense_rows) {
      exception_values[0] = 3;
      exception_values[1] = i;
      exception_values[2] = row;
      exception_values[3] = dense_rows;
      return 0;
    }
    ++csr_offset[row];
    rows_are_ordered = rows_are_ordered && (row >= last_indices_row);
    last_indices_row = row;
  }

  bool all_rows_full = true;
  for (size_t row = 0; row < dense_rows; ++row) {
    // csr_offset here describes the number of elements in this dense row
    const bool row_empty = (csr_offset[row] == 0);
    empty_row_indicator[row] = row_empty;
    all_rows_full = all_rows_full && !row_empty;
    // In filled version, each row has at least one element.
    csr_offset[row] = std::max<size_t>(csr_offset[row], 1);
    // Update csr_offset to represent the number of elements up to and
    // including dense_rows + 1:
    //  csr_offset[0] == #{elements of row 0}
    //  csr_offset[1] == #{elements of row 1} + #{elements of row 0}
    //  ..
    //  csr_offset[i] == starting index for elements in row i + 1.
    if (row > 0) {
      csr_offset[row] += csr_offset[row - 1];
    }
  }

  if (all_rows_full && rows_are_ordered) {
    std::copy(indices, indices + indices_count * rank, output_indices);
    std::copy(values, values + indices_count, output_values);
    std::iota(reverse_index_map, reverse_index_map + indices_count, 0);
    return indices_count;
  } else {
    const size_t full_indices_count = csr_offset[dense_rows - 1];
    std::vector<size_t> filled_count(dense_rows, 0 * full_indices_count);

    // Fill in values for rows that are not missing
    for (size_t i = 0; i < indices_count; ++i) {
      // indices is a 2d tensor with shape of [N, rank]
      const int32_t row = indices[i * rank];
      const size_t offset = filled_count[row];
      const size_t output_i = ((row == 0) ? 0 : csr_offset[row - 1]) + offset;
      filled_count[row]++;  // Increment the filled count for this row.
      for (size_t j = 0; j < rank; ++j) {
        // indices and output_indices are 2d tensors with shape of [_, rank]
        output_indices[output_i * rank + j] = indices[i * rank + j];
      }
      output_values[output_i] = values[i];
      // We'll need this reverse index map to backprop correctly.
      reverse_index_map[i] = output_i;
    }

    // Fill in values for rows that are missing
    for (size_t row = 0; row < dense_rows; ++row) {
      const size_t row_count = filled_count[row];
      if (row_count == 0) {  // We haven't filled this row
        const size_t starting_index = (row == 0) ? 0 : csr_offset[row - 1];
        // Remaining index values were set to zero already.
        // Just need to set the row index in the right location.
        // output_indices is a 2d tensor with shape of [_, rank]
        output_indices[starting_index * rank + 0] = row;
        for (size_t col = 1; col < rank; ++col) {
          output_indices[starting_index * rank + col] = 0;
        }
        output_values[starting_index] = default_value;
      }
    }
    return full_indices_count;
  }
}
}  // namespace

namespace tfjs {
namespace wasm {
extern "C" {
#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

size_t SparseFillEmptyRows(
    const size_t indices_id, const size_t values_id, const DType values_dtype,
    const size_t indices_count, const size_t dense_rows, const size_t rank,
    const size_t default_value_id, const size_t output_indices_id,
    const size_t output_values_id, const size_t empty_row_indicator_id,
    const size_t reverse_index_map_id, const size_t exception_values_id) {
  auto& indices_info = backend::get_tensor_info(indices_id);
  auto& values_info = backend::get_tensor_info(values_id);
  auto& default_value_info = backend::get_tensor_info(default_value_id);
  auto& output_indices_info = backend::get_tensor_info_out(output_indices_id);
  auto& output_values_info = backend::get_tensor_info_out(output_values_id);
  auto& empty_row_indicator_info =
      backend::get_tensor_info_out(empty_row_indicator_id);
  auto& reverse_index_map_info =
      backend::get_tensor_info_out(reverse_index_map_id);
  auto& exception_values_info =
      backend::get_tensor_info_out(exception_values_id);

  const int32_t* indices = indices_info.i32();
  int32_t* output_indices = output_indices_info.i32_write();
  bool* empty_row_indicator = empty_row_indicator_info.b_write();
  int32_t* reverse_index_map = reverse_index_map_info.i32_write();
  int32_t* exception_values = exception_values_info.i32_write();

  switch (values_dtype) {
    case DType::float32:
      return sparse_fill_empty_rows<float>(
          indices, indices_count, rank, values_info.f32(), dense_rows,
          default_value_info.f32()[0], output_indices,
          output_values_info.f32_write(), empty_row_indicator,
          reverse_index_map, exception_values);
    case DType::int32:
      return sparse_fill_empty_rows<int32_t>(
          indices, indices_count, rank, values_info.i32(), dense_rows,
          default_value_info.i32()[0], output_indices,
          output_values_info.i32_write(), empty_row_indicator,
          reverse_index_map, exception_values);
    default:
      util::warn("SparseFillEmptyRows failed. Unknown dtype %d", values_dtype);
      return 0;
  }
}
}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
