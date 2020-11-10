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

#ifndef NON_MAX_SUPPRESSION_IMPL_H_
#define NON_MAX_SUPPRESSION_IMPL_H_

#include <cstddef>
#include <vector>

namespace tfjs {
namespace wasm {

// Structure to store the result of the kernel. In this case we give js a
// a pointer in memory where the result is stored and how big it is.
struct NonMaxSuppressionResult {
  int32_t* selected_indices;
  size_t selected_size;
  float* selected_scores;
  size_t* valid_outputs;
};

const NonMaxSuppressionResult* non_max_suppression_impl(
    const size_t boxes_id, const size_t scores_id, const size_t max_out_size,
    const float iou_threshold, const float score_threshold,
    const float soft_nms_sigma, const bool pad_to_max_output_size);

}  // namespace wasm
}  // namespace tfjs

#endif  // NON_MAX_SUPPRESSION_IMPL_H_
