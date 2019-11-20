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

#include <algorithm>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "src/cc/backend.h"
#include "src/cc/util.h"

namespace {

float compute_iou(const float* boxes, const int i, const int j) { return 0.0; }

struct Result {
  int* buf;
  int size;
};

}  // namespace

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
const Result* NonMaxSuppressionV3(const int boxes_id, const int scores_id,
                                  const int max_out_size,
                                  const float iou_threshold,
                                  const float score_threshold) {
  auto& boxes_info = backend::get_tensor_info(boxes_id);
  auto& scores_info = backend::get_tensor_info_out(scores_id);
  const float* boxes = boxes_info.f32();
  const float* scores = scores_info.f32();
  const int num_boxes = boxes_info.size / 4;
  std::vector<int> box_indices;
  for (size_t i = 0; i < num_boxes; ++i) {
    if (scores[i] > score_threshold) {
      box_indices.push_back(i);
    }
  }
  // Sort by scores.
  std::sort(
      box_indices.begin(), box_indices.end(),
      [&scores](const int i, const int j) { return scores[i] > scores[j]; });

  std::vector<int> selected;
  for (size_t i = 0; i < box_indices.size(); ++i) {
    const int box_i = box_indices[i];
    bool ignore_candidate = false;
    for (size_t j = 0; j < selected.size(); ++j) {
      const int box_j = selected[j];
      const float iou = compute_iou(boxes, box_i, box_j);
      if (iou >= iou_threshold) {
        ignore_candidate = true;
        break;
      }
    }
    if (!ignore_candidate) {
      selected.push_back(box_i);
      if (selected.size() >= max_out_size) {
        break;
      }
    }
  }
  int* data = static_cast<int*>(malloc(selected.size() * sizeof(int)));
  std::memcpy(data, selected.data(), selected.size() * sizeof(int));
  return new Result{data, static_cast<int>(selected.size())};
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
