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

import {TypedArray} from '../types';
import {binaryInsert} from './non_max_suppression_util';

/**
 * Implementation of the NonMaxSuppression kernel shared between webgl and cpu.
 */
interface Candidate {
  score: number;
  boxIndex: number;
  suppressBeginIndex: number;
}

interface NonMaxSuppressionResult {
  selectedIndices: number[];
  selectedScores?: number[];
  validOutputs?: number;
}

export function nonMaxSuppressionV3Impl(
    boxes: TypedArray, scores: TypedArray, maxOutputSize: number,
    iouThreshold: number, scoreThreshold: number): NonMaxSuppressionResult {
  return nonMaxSuppressionImpl_(
      boxes, scores, maxOutputSize, iouThreshold, scoreThreshold,
      0 /* softNmsSigma */);
}

export function nonMaxSuppressionV4Impl(
    boxes: TypedArray, scores: TypedArray, maxOutputSize: number,
    iouThreshold: number, scoreThreshold: number,
    padToMaxOutputSize: boolean): NonMaxSuppressionResult {
  return nonMaxSuppressionImpl_(
      boxes, scores, maxOutputSize, iouThreshold, scoreThreshold,
      0 /* softNmsSigma */, false /* returnScoresTensor */,
      padToMaxOutputSize /* padToMaxOutputSize */, true
      /* returnValidOutputs */);
}

export function nonMaxSuppressionV5Impl(
    boxes: TypedArray, scores: TypedArray, maxOutputSize: number,
    iouThreshold: number, scoreThreshold: number,
    softNmsSigma: number): NonMaxSuppressionResult {
  return nonMaxSuppressionImpl_(
      boxes, scores, maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma,
      true /* returnScoresTensor */);
}

function nonMaxSuppressionImpl_(
    boxes: TypedArray, scores: TypedArray, maxOutputSize: number,
    iouThreshold: number, scoreThreshold: number, softNmsSigma: number,
    returnScoresTensor = false, padToMaxOutputSize = false,
    returnValidOutputs = false): NonMaxSuppressionResult {
  // The list is sorted in ascending order, so that we can always pop the
  // candidate with the largest score in O(1) time.
  const candidates = [];

  for (let i = 0; i < scores.length; i++) {
    if (scores[i] > scoreThreshold) {
      candidates.push({score: scores[i], boxIndex: i, suppressBeginIndex: 0});
    }
  }

  candidates.sort(ascendingComparator);

  // If softNmsSigma is 0, the outcome of this algorithm is exactly same as
  // before.
  const scale = softNmsSigma > 0 ? (-0.5 / softNmsSigma) : 0.0;

  const selectedIndices: number[] = [];
  const selectedScores: number[] = [];

  while (selectedIndices.length < maxOutputSize && candidates.length > 0) {
    const candidate = candidates.pop();
    const {score: originalScore, boxIndex, suppressBeginIndex} = candidate;

    if (originalScore < scoreThreshold) {
      break;
    }

    // Overlapping boxes are likely to have similar scores, therefore we
    // iterate through the previously selected boxes backwards in order to
    // see if candidate's score should be suppressed. We use
    // suppressBeginIndex to track and ensure a candidate can be suppressed
    // by a selected box no more than once. Also, if the overlap exceeds
    // iouThreshold, we simply ignore the candidate.
    let ignoreCandidate = false;
    for (let j = selectedIndices.length - 1; j >= suppressBeginIndex; --j) {
      const iou = intersectionOverUnion(boxes, boxIndex, selectedIndices[j]);

      if (iou >= iouThreshold) {
        ignoreCandidate = true;
        break;
      }

      candidate.score =
          candidate.score * suppressWeight(iouThreshold, scale, iou);

      if (candidate.score <= scoreThreshold) {
        break;
      }
    }

    // At this point, if `candidate.score` has not dropped below
    // `scoreThreshold`, then we know that we went through all of the
    // previous selections and can safely update `suppressBeginIndex` to the
    // end of the selected array. Then we can re-insert the candidate with
    // the updated score and suppressBeginIndex back in the candidate list.
    // If on the other hand, `candidate.score` has dropped below the score
    // threshold, we will not add it back to the candidates list.
    candidate.suppressBeginIndex = selectedIndices.length;

    if (!ignoreCandidate) {
      // Candidate has passed all the tests, and is not suppressed, so
      // select the candidate.
      if (candidate.score === originalScore) {
        selectedIndices.push(boxIndex);
        selectedScores.push(candidate.score);
      } else if (candidate.score > scoreThreshold) {
        // Candidate's score is suppressed but is still high enough to be
        // considered, so add back to the candidates list.
        binaryInsert(candidates, candidate, ascendingComparator);
      }
    }
  }

  // NonMaxSuppressionV4 feature: padding output to maxOutputSize.
  const validOutputs = selectedIndices.length;
  const elemsToPad = maxOutputSize - validOutputs;

  if (padToMaxOutputSize && elemsToPad > 0) {
    selectedIndices.push(...new Array(elemsToPad).fill(0));
    selectedScores.push(...new Array(elemsToPad).fill(0.0));
  }

  const result: NonMaxSuppressionResult = {selectedIndices};

  if (returnScoresTensor) {
    result['selectedScores'] = selectedScores;
  }

  if (returnValidOutputs) {
    result['validOutputs'] = validOutputs;
  }

  return result;
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

// A Gaussian penalty function, this method always returns values in [0, 1].
// The weight is a function of similarity, the more overlap two boxes are, the
// smaller the weight is, meaning highly overlapping boxe will be significantly
// penalized. On the other hand, a non-overlapping box will not be penalized.
function suppressWeight(iouThreshold: number, scale: number, iou: number) {
  const weight = Math.exp(scale * iou * iou);
  return iou <= iouThreshold ? weight : 0.0;
}

function ascendingComparator(c1: Candidate, c2: Candidate) {
  // For objects with same scores, we make the object with the larger index go
  // first. In an array that pops from the end, this means that the object with
  // the smaller index will be popped first. This ensures the same output as
  // the TensorFlow python version.
  return (c1.score - c2.score) ||
      ((c1.score === c2.score) && (c2.boxIndex - c1.boxIndex));
}
