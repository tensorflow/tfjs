/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
 * =============================================================================
 */

import {ENGINE} from '../../engine';
import {NonMaxSuppressionV3} from '../../kernel_names';
import {Tensor1D, Tensor2D} from '../../tensor';
import {convertToTensor} from '../../tensor_util_env';
import {TensorLike} from '../../types';

import {nonMaxSuppSanityCheck} from '../nonmax_util';
import {op} from '../operation';

function nonMaxSuppression_(
    boxes: Tensor2D|TensorLike, scores: Tensor1D|TensorLike,
    maxOutputSize: number, iouThreshold = 0.5,
    scoreThreshold = Number.NEGATIVE_INFINITY): Tensor1D {
  const $boxes = convertToTensor(boxes, 'boxes', 'nonMaxSuppression');
  const $scores = convertToTensor(scores, 'scores', 'nonMaxSuppression');

  const inputs = nonMaxSuppSanityCheck(
      $boxes, $scores, maxOutputSize, iouThreshold, scoreThreshold);
  maxOutputSize = inputs.maxOutputSize;
  iouThreshold = inputs.iouThreshold;
  scoreThreshold = inputs.scoreThreshold;

  const attrs = {maxOutputSize, iouThreshold, scoreThreshold};
  return ENGINE.runKernelFunc(
      b => b.nonMaxSuppression(
          $boxes, $scores, maxOutputSize, iouThreshold, scoreThreshold),
      {boxes: $boxes, scores: $scores}, null /* grad */, NonMaxSuppressionV3,
      attrs);
}

export const nonMaxSuppression = op({nonMaxSuppression_});
