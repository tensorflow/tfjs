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
#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

#include "tfjs-backend-wasm/src/cc/non_max_suppression_impl.h"

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
const NonMaxSuppressionResult* NonMaxSuppressionV3(
    const size_t boxes_id, const size_t scores_id, const size_t max_out_size,
    const float iou_threshold, const float score_threshold) {
  return tfjs::wasm::non_max_suppression_impl(
      boxes_id, scores_id, max_out_size, iou_threshold, score_threshold,
      0.0 /* soft_nms_sigma */, false /* pad_to_max_output_size */);
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
