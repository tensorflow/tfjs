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

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <deque>
#include <memory>
#include <queue>
#include <vector>

#include "src/cc/backend.h"
#include "src/cc/non_max_suppression_impl.h"

namespace {

struct Candidate {
  int32_t box_index;
  float score;
  int32_t suppress_begin_index;
};

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

float suppress_weight(const float iou_threshold, const float scale,
                      const float iou) {
  const float weight = std::exp(scale * iou * iou);
  return iou <= iou_threshold ? weight : 0.0;
}
}  // namespace

namespace tfjs {
namespace wasm {
const NonMaxSuppressionResult* non_max_suppression_impl(
    const size_t boxes_id, const size_t scores_id, const size_t max_out_size,
    const float iou_threshold, const float score_threshold,
    const float soft_nms_sigma, const bool pad_to_max_output_size) {
  auto& boxes_info = backend::get_tensor_info(boxes_id);
  auto& scores_info = backend::get_tensor_info_out(scores_id);
  const float* boxes = boxes_info.f32();
  const float* scores = scores_info.f32();
  const size_t num_boxes = boxes_info.size / 4;

  auto score_comparator = [](const Candidate i, const Candidate j) {
    return i.score < j.score ||
           ((i.score == j.score) && (i.box_index > j.box_index));
  };
  // Construct a max heap by candidate scores.
  std::priority_queue<Candidate, std::deque<Candidate>,
                      decltype(score_comparator)>
      candidate_priority_queue(score_comparator);

  const int32_t suppress_at_start = 0;
  // Filter out boxes that are below the score threshold and also maintain
  // the order of boxes by scores.
  for (int32_t i = 0; i < num_boxes; i++) {
    if (scores[i] > score_threshold) {
      candidate_priority_queue.emplace(
          Candidate({i, scores[i], suppress_at_start}));
    }
  }

  // If soft_nms_sigma is 0, the outcome of this algorithm is exactly same as
  // before.
  const float scale = soft_nms_sigma > 0.0 ? (-0.5 / soft_nms_sigma) : 0.0;

  // Select a box only if it doesn't overlap beyond the threshold with the
  // already selected boxes.
  std::vector<int32_t> selected_indices;
  std::vector<float> selected_scores;
  Candidate candidate;
  float iou, original_score;

  while (selected_indices.size() < max_out_size &&
         !candidate_priority_queue.empty()) {
    candidate = candidate_priority_queue.top();
    original_score = candidate.score;
    candidate_priority_queue.pop();

    if (original_score < score_threshold) {
      break;
    }

    // Overlapping boxes are likely to have similar scores, therefore we
    // iterate through the previously selected boxes backwards in order to
    // see if candidate's score should be suppressed. We use
    // suppress_begin_index to track and ensure a candidate can be suppressed
    // by a selected box no more than once. Also, if the overlap exceeds
    // iou_threshold, we simply ignore the candidate.
    bool ignore_candidate = false;
    for (int32_t j = selected_indices.size() - 1;
         j >= candidate.suppress_begin_index; --j) {
      const float iou =
          compute_iou(boxes, candidate.box_index, selected_indices[j]);

      if (iou >= iou_threshold) {
        ignore_candidate = true;
        break;
      }

      candidate.score *= suppress_weight(iou_threshold, scale, iou);

      if (candidate.score <= score_threshold) {
        break;
      }
    }

    // At this point, if `candidate.score` has not dropped below
    // `score_threshold`, then we know that we went through all of the
    // previous selections and can safely update `suppress_begin_index` to the
    // end of the selected array. Then we can re-insert the candidate with
    // the updated score and suppress_begin_index back in the candidate queue.
    // If on the other hand, `candidate.score` has dropped below the score
    // threshold, we will not add it back to the candidates queue.
    candidate.suppress_begin_index = selected_indices.size();

    if (!ignore_candidate) {
      // Candidate has passed all the tests, and is not suppressed, so
      // select the candidate.
      if (candidate.score == original_score) {
        selected_indices.push_back(candidate.box_index);
        selected_scores.push_back(candidate.score);
      } else if (candidate.score > score_threshold) {
        // Candidate's score is suppressed but is still high enough to be
        // considered, so add back to the candidates queue.
        candidate_priority_queue.push(candidate);
      }
    }
  }

  size_t num_valid_outputs = selected_indices.size();
  if (pad_to_max_output_size) {
    selected_indices.resize(max_out_size, 0);
    selected_scores.resize(max_out_size, 0.0);
  }

  // Allocate memory on the heap for the results and copy the data from the
  // `selected_indices` and `selected_scores` vector since we can't "steal" the
  // data from the vector.
  size_t selected_indices_data_size = selected_indices.size() * sizeof(int32_t);
  int32_t* selected_indices_data =
      static_cast<int32_t*>(malloc(selected_indices_data_size));
  std::memcpy(selected_indices_data, selected_indices.data(),
              selected_indices_data_size);

  size_t selected_scores_data_size = selected_scores.size() * sizeof(float);
  float* selected_scores_data =
      static_cast<float*>(malloc(selected_scores_data_size));
  std::memcpy(selected_scores_data, selected_scores.data(),
              selected_scores_data_size);

  size_t valid_outputs_data_size = sizeof(size_t);
  size_t* valid_outputs_data =
      static_cast<size_t*>(malloc(valid_outputs_data_size));
  *valid_outputs_data = num_valid_outputs;

  // Allocate the result of the method on the heap so it survives past this
  // function and we can read it in js.
  return new NonMaxSuppressionResult{selected_indices_data,
                                     selected_indices.size(),
                                     selected_scores_data, valid_outputs_data};
}

}  // namespace wasm
}  // namespace tfjs
