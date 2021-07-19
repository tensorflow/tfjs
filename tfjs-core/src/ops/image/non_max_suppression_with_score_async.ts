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
import {nonMaxSuppressionV5Impl} from '../../backends/non_max_suppression_impl';
import {Tensor1D, Tensor2D} from '../../tensor';
import {NamedTensorMap} from '../../tensor_types';
import {convertToTensor} from '../../tensor_util_env';
import {TensorLike} from '../../types';
import {nonMaxSuppSanityCheck} from '../nonmax_util';
import {tensor1d} from '../tensor1d';

/**
 * Asynchronously performs non maximum suppression of bounding boxes based on
 * iou (intersection over union).
 *
 * This op also supports a Soft-NMS mode (c.f.
 * Bodla et al, https://arxiv.org/abs/1704.04503) where boxes reduce the score
 * of other overlapping boxes, therefore favoring different regions of the image
 * with high scores. To enable this Soft-NMS mode, set the `softNmsSigma`
 * parameter to be larger than 0.
 *
 * @param boxes a 2d tensor of shape `[numBoxes, 4]`. Each entry is
 *     `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the corners of
 *     the bounding box.
 * @param scores a 1d tensor providing the box scores of shape `[numBoxes]`.
 * @param maxOutputSize The maximum number of boxes to be selected.
 * @param iouThreshold A float representing the threshold for deciding whether
 *     boxes overlap too much with respect to IOU. Must be between [0, 1].
 *     Defaults to 0.5 (50% box overlap).
 * @param scoreThreshold A threshold for deciding when to remove boxes based
 *     on score. Defaults to -inf, which means any score is accepted.
 * @param softNmsSigma A float representing the sigma parameter for Soft NMS.
 *     When sigma is 0, it falls back to nonMaxSuppression.
 * @return A map with the following properties:
 *     - selectedIndices: A 1D tensor with the selected box indices.
 *     - selectedScores: A 1D tensor with the corresponding scores for each
 *       selected box.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
async function nonMaxSuppressionWithScoreAsync_(
    boxes: Tensor2D|TensorLike, scores: Tensor1D|TensorLike,
    maxOutputSize: number, iouThreshold = 0.5,
    scoreThreshold = Number.NEGATIVE_INFINITY,
    softNmsSigma = 0.0): Promise<NamedTensorMap> {
  const $boxes = convertToTensor(boxes, 'boxes', 'nonMaxSuppressionAsync');
  const $scores = convertToTensor(scores, 'scores', 'nonMaxSuppressionAsync');

  const params = nonMaxSuppSanityCheck(
      $boxes, $scores, maxOutputSize, iouThreshold, scoreThreshold,
      softNmsSigma);
  maxOutputSize = params.maxOutputSize;
  iouThreshold = params.iouThreshold;
  scoreThreshold = params.scoreThreshold;
  softNmsSigma = params.softNmsSigma;

  const boxesAndScores = await Promise.all([$boxes.data(), $scores.data()]);
  const boxesVals = boxesAndScores[0];
  const scoresVals = boxesAndScores[1];

  // We call a cpu based impl directly with the typedarray data  here rather
  // than a kernel because all kernels are synchronous (and thus cannot await
  // .data()).
  const {selectedIndices, selectedScores} = nonMaxSuppressionV5Impl(
      boxesVals, scoresVals, maxOutputSize, iouThreshold, scoreThreshold,
      softNmsSigma);

  if ($boxes !== boxes) {
    $boxes.dispose();
  }
  if ($scores !== scores) {
    $scores.dispose();
  }

  return {
    selectedIndices: tensor1d(selectedIndices, 'int32'),
    selectedScores: tensor1d(selectedScores)
  };
}

export const nonMaxSuppressionWithScoreAsync = nonMaxSuppressionWithScoreAsync_;
