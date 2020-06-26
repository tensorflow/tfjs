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

import {Tensor1D, Tensor2D} from '../tensor';
import * as util from '../util';

function nonMaxSuppSanityCheck(
    boxes: Tensor2D, scores: Tensor1D, maxOutputSize: number,
    iouThreshold: number, scoreThreshold: number, softNmsSigma?: number): {
  maxOutputSize: number,
  iouThreshold: number,
  scoreThreshold: number,
  softNmsSigma: number
} {
  if (iouThreshold == null) {
    iouThreshold = 0.5;
  }
  if (scoreThreshold == null) {
    scoreThreshold = Number.NEGATIVE_INFINITY;
  }
  if (softNmsSigma == null) {
    softNmsSigma = 0.0;
  }

  const numBoxes = boxes.shape[0];
  maxOutputSize = Math.min(maxOutputSize, numBoxes);

  util.assert(
      0 <= iouThreshold && iouThreshold <= 1,
      () => `iouThreshold must be in [0, 1], but was '${iouThreshold}'`);
  util.assert(
      boxes.rank === 2,
      () => `boxes must be a 2D tensor, but was of rank '${boxes.rank}'`);
  util.assert(
      boxes.shape[1] === 4,
      () =>
          `boxes must have 4 columns, but 2nd dimension was ${boxes.shape[1]}`);
  util.assert(scores.rank === 1, () => 'scores must be a 1D tensor');
  util.assert(
      scores.shape[0] === numBoxes,
      () => `scores has incompatible shape with boxes. Expected ${numBoxes}, ` +
          `but was ${scores.shape[0]}`);
  util.assert(
      0 <= softNmsSigma && softNmsSigma <= 1,
      () => `softNmsSigma must be in [0, 1], but was '${softNmsSigma}'`);
  return {maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma};
}

export {nonMaxSuppSanityCheck};
