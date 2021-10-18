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

namespace {

template <typename T>
void sparse_segment_reduction(const T* input, const size_t input_length,
                              const size_t num_row, const int32_t* indices,
                              const size_t num_indices,
                              const int32_t* segment_ids, T* output,
                              size_t output_length, int32_t* exception_values,
                              const bool is_mean, const int32_t default_value) {
  // Initialize to no exceptions.
  exception_values[0] = -1;

  // Flatten the array to two dimensions
  const size_t input_flat[] = {num_row, input_length / num_row};
  const size_t num_col = input_flat[1];
  // Note that the current implementation assumes that segment_ids values are
  // sorted.
  const int32_t last_segment_id_plus_one =
      num_indices > 0 ? segment_ids[num_indices - 1] + 1 : 0;
  const int32_t output_rows = last_segment_id_plus_one;

  if (output_rows < 0) {
    exception_values[0] = 0;
    return;
  }

  // Note that we do not initialize the output buffer with a default value, so
  // we need to explicitly set missing indices to the default value.
  if (num_indices == 0) {
    if (output_rows > 0) {
      std::fill(output, output + output_length, default_value);
    }
    return;
  }

  if (output_rows <= 0) {
    exception_values[0] = 0;
    return;
  }

  std::fill(output, output + output_length, 0);

  int32_t start = 0, end = 1;
  // Index from which the output is not initialized.
  int32_t uninitialized_index = 0;
  int32_t out_index = segment_ids[start];

  while (true) {
    // We initialize next_index to 0 to avoid may be uninitialized warning
    int32_t next_index = 0;
    if (end < num_indices) {
      next_index = segment_ids[end];
      if (out_index == next_index) {
        ++end;
        continue;
      }
      // We have a new segment here.  Verify that the segment ids are growing.
      if (out_index >= next_index) {
        exception_values[0] = 1;
        return;
      }
    }

    if (out_index < 0 || out_index >= output_rows) {
      exception_values[0] = 2;
      exception_values[1] = out_index;
      exception_values[2] = output_rows;
      return;
    }

    // If there is a gap between two indices, we need to set that gap to the
    // default value.
    if (out_index > uninitialized_index) {
      std::fill(output + uninitialized_index * num_col,
                output + out_index * num_col, default_value);
    }

    for (int32_t i = start; i < end; ++i) {
      const int32_t index = indices[i];
      if (index < 0 || index >= input_flat[0]) {
        exception_values[0] = 3;
        exception_values[1] = i;
        exception_values[2] = indices[i];
        exception_values[3] = input_flat[0];
      }
      for (int32_t j = 0; j < num_col; j++) {
        output[out_index * num_col + j] += input[index * num_col + j];
      }
    }

    if (is_mean) {
      for (size_t j = 0; j < num_col; j++) {
        output[out_index * num_col + j] /= end - start;
      }
    }

    start = end;
    ++end;
    uninitialized_index = out_index + 1;
    out_index = next_index;
    if (end > num_indices) {
      break;
    }
  }

  // Fill the gap at the end with the default value.
  if (uninitialized_index < output_rows) {
    std::fill(output + uninitialized_index * num_col,
              output + output_rows * num_col, default_value);
  }
}
}  // namespace

namespace tfjs {
namespace wasm {
extern "C" {
#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif

void SparseSegmentReduction(const size_t input_id, const DType dtype,
                            const size_t num_row, const size_t indices_id,
                            const size_t segment_ids_id, const size_t output_id,
                            const size_t exception_values_id,
                            const bool is_mean, const int32_t default_value) {
  auto& input_info = backend::get_tensor_info(input_id);
  auto& indices_info = backend::get_tensor_info(indices_id);
  auto& segment_ids_info = backend::get_tensor_info_out(segment_ids_id);
  auto& output_info = backend::get_tensor_info_out(output_id);
  auto& exception_values_info =
      backend::get_tensor_info_out(exception_values_id);

  const int32_t* indices = indices_info.i32();
  const int32_t* segment_ids = segment_ids_info.i32();
  int32_t* exception_values = exception_values_info.i32_write();

  switch (dtype) {
    case DType::float32:
      sparse_segment_reduction<float>(
          input_info.f32(), input_info.size, num_row, indices,
          indices_info.size, segment_ids, output_info.f32_write(),
          output_info.size, exception_values, is_mean, default_value);
      break;
    case DType::int32:
      sparse_segment_reduction<int32_t>(
          input_info.i32(), input_info.size, num_row, indices,
          indices_info.size, segment_ids, output_info.i32_write(),
          output_info.size, exception_values, is_mean, default_value);
      break;
    default:
      util::warn("SparseSegmentReduction failed. Unknown dtype %d", dtype);
  }
}
}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
