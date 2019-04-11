/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

/**
 * Implementation of the NonMaxSuppression kernel shared between webgl and cpu.
 */

import {tensor1d} from '../ops/tensor_ops';
import {Tensor1D} from '../tensor';
import {TypedArray} from '../types';

export function nonMaxSuppressionImpl(
    boxes: TypedArray, scores: TypedArray, maxOutputSize: number,
    iouThreshold: number, scoreThreshold: number): Tensor1D {
  const candidates = Array.from(scores)
                         .map((score, boxIndex) => ({score, boxIndex}))
                         .filter(c => c.score > scoreThreshold)
                         .sort((c1, c2) => c2.score - c1.score);

  const selected: number[] = [];

  for (let i = 0; i < candidates.length; i++) {
    const {score, boxIndex} = candidates[i];
    if (score < scoreThreshold) {
      break;
    }

    let ignoreCandidate = false;
    for (let j = selected.length - 1; j >= 0; --j) {
      const iou = intersectionOverUnion(boxes, boxIndex, selected[j]);
      if (iou >= iouThreshold) {
        ignoreCandidate = true;
        break;
      }
    }

    if (!ignoreCandidate) {
      selected.push(boxIndex);
      if (selected.length >= maxOutputSize) {
        break;
      }
    }
  }

  return tensor1d(selected, 'int32');
}

function intersectionOverUnion(boxes: TypedArray, i: number, j: number) {
  const iCoord = boxes.subarray(i * 4, i * 4 + 4);
  const jCoord = boxes.subarray(j * 4, j * 4 + 4);
  const yminI = Math.min(iCoord[0], iCoord[2]);
  const xminI = Math.min(iCoord[1], iCoord[3]);
  const ymaxI = Math.max(iCoord[0], iCoord[2]);
  const xmaxI = Math.max(iCoord[1], iCoord[3]);
  const yminJ = Math.min(jCoord[0], jCoord[2]);
  const xminJ = Math.min(jCoord[1], jCoord[3]);
  const ymaxJ = Math.max(jCoord[0], jCoord[2]);
  const xmaxJ = Math.max(jCoord[1], jCoord[3]);
  const areaI = (ymaxI - yminI) * (xmaxI - xminI);
  const areaJ = (ymaxJ - yminJ) * (xmaxJ - xminJ);
  if (areaI <= 0 || areaJ <= 0) {
    return 0.0;
  }
  const intersectionYmin = Math.max(yminI, yminJ);
  const intersectionXmin = Math.max(xminI, xminJ);
  const intersectionYmax = Math.min(ymaxI, ymaxJ);
  const intersectionXmax = Math.min(xmaxI, xmaxJ);
  const intersectionArea = Math.max(intersectionYmax - intersectionYmin, 0.0) *
      Math.max(intersectionXmax - intersectionXmin, 0.0);
  return intersectionArea / (areaI + areaJ - intersectionArea);
}
