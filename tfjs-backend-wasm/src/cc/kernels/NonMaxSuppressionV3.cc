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
#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

#include "src/cc/backend.h"
#include "src/cc/util.h"

namespace {

float compute_iou(const float* boxes, const size_t i, const size_t j) {
  const float* i_coord = boxes + i * 4;
  const float* j_coord = boxes + j * 4;

  const float y_min_i = std::min(i_coord[0], i_coord[2]);
  const float x_min_i = std::min(i_coord[1], i_coord[3]);

  const float y_max_i = std::max(i_coord[0], i_coord[2]);
  const float x_max_i = std::max(i_coord[1], i_coord[3]);

  const float y_min_j = std::min(j_coord[0], j_coord[2]);
  const float x_min_j = std::min(j_coord[1], j_coord[3]);

  const float y_max_j = std::max(j_coord[0], j_coord[2]);
  const float x_max_j = std::max(j_coord[1], j_coord[3]);

  const float area_i = (y_max_i - y_min_i) * (x_max_i - x_min_i);
  const float area_j = (y_max_j - y_min_j) * (x_max_j - x_min_j);

  if (area_i <= 0 || area_j <= 0) {
    return 0.0;
  }

  const float intersect_y_min = std::max(y_min_i, y_min_j);
  const float intersect_x_min = std::max(x_min_i, x_min_j);
  const float intersect_y_max = std::min(y_max_i, y_max_j);
  const float intersect_x_max = std::min(x_max_i, x_max_j);
  const float intersect_area =
      std::max(intersect_y_max - intersect_y_min, .0f) *
      std::max(intersect_x_max - intersect_x_min, .0f);
  return intersect_area / (area_i + area_j - intersect_area);
}

// Structure to store the result of the kernel. In this case we give js a
// a pointer in memory where the result is stored and how big it is.
struct Result {
  int32_t* buf;
  size_t size;
};

}  // namespace

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
const Result* NonMaxSuppressionV3(const size_t boxes_id, const size_t scores_id,
                                  const size_t max_out_size,
                                  const float iou_threshold,
                                  const float score_threshold) {
  auto& boxes_info = backend::get_tensor_info(boxes_id);
  auto& scores_info = backend::get_tensor_info_out(scores_id);
  const float* boxes = boxes_info.f32();
  const float* scores = scores_info.f32();
  const size_t num_boxes = boxes_info.size / 4;

  // Filter out boxes that are below the score threshold.
  std::vector<int32_t> box_indices;
  for (int32_t i = 0; i < num_boxes; ++i) {
    if (scores[i] > score_threshold) {
      box_indices.push_back(i);
    }
  }

  // Sort by remaining boxes by scores.
  std::sort(box_indices.begin(), box_indices.end(),
            [&scores](const size_t i, const size_t j) {
              return scores[i] > scores[j];
            });

  // Select a box only if it doesn't overlap beyond the threshold with the
  // already selected boxes.
  std::vector<int32_t> selected;
  for (int32_t i = 0; i < box_indices.size(); ++i) {
    const size_t box_i = box_indices[i];
    bool ignore_candidate = false;
    for (int32_t j = 0; j < selected.size(); ++j) {
      const int32_t box_j = selected[j];
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

  // Allocate memory on the heap for the resulting indices and copy the data
  // from the `selected` vector since we can't "steal" the data from the
  // vector.
  int32_t* data =
      static_cast<int32_t*>(malloc(selected.size() * sizeof(int32_t)));
  std::memcpy(data, selected.data(), selected.size() * sizeof(int32_t));

  // Allocate the result of the method on the heap so it survives past this
  // function and we can read it in js.
  return new Result{data, selected.size()};
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
