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

import {Tensor1D, Tensor2D} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {cast} from './cast';
import {matMul} from './mat_mul';
import {oneHot} from './one_hot';
import {op} from './operation';
import {transpose} from './transpose';

/**
 * Computes the confusion matrix from true labels and predicted labels.
 *
 * ```js
 * const labels = tf.tensor1d([0, 1, 2, 1, 0], 'int32');
 * const predictions = tf.tensor1d([0, 2, 2, 1, 0], 'int32');
 * const numClasses = 3;
 * const out = tf.math.confusionMatrix(labels, predictions, numClasses);
 * out.print();
 * // Expected output matrix:
 * // [[2, 0, 0],
 * //  [0, 1, 1],
 * //  [0, 0, 1]]
 * ```
 *
 * @param labels The target labels, assumed to be 0-based integers
 *   for the classes. The shape is `[numExamples]`, where
 *   `numExamples` is the number of examples included.
 * @param predictions The predicted classes, assumed to be
 *   0-based integers for the classes. Must have the same shape as `labels`.
 * @param numClasses Number of all classes, as an integer.
 *   Its value must be larger than the largest element in `labels` and
 *   `predictions`.
 * @returns The confusion matrix as a int32-type 2D tensor. The value at
 *   row `r` and column `c` is the number of times examples of actual class
 *   `r` were predicted as class `c`.
 */
/** @doc {heading: 'Operations', subheading: 'Evaluation'} */
export function confusionMatrix_(
    labels: Tensor1D|TensorLike, predictions: Tensor1D|TensorLike,
    numClasses: number): Tensor2D {
  const $labels = convertToTensor(labels, 'labels', 'confusionMatrix');
  const $predictions =
      convertToTensor(predictions, 'predictions', 'confusionMatrix');

  util.assert(
      numClasses == null || numClasses > 0 && Number.isInteger(numClasses),
      () => `If provided, numClasses must be a positive integer, ` +
          `but got ${numClasses}`);
  util.assert(
      $labels.rank === 1,
      () => `Expected the rank of labels to be 1, but got ${$labels.rank}`);
  util.assert(
      $predictions.rank === 1,
      () => `Expected the rank of predictions to be 1, ` +
          `but got ${$predictions.rank}`);
  util.assert(
      $labels.shape[0] === $predictions.shape[0],
      () => `Mismatch in the number of examples: ` +
          `${$labels.shape[0]} vs. ${$predictions.shape[0]}. ` +
          `Labels and predictions should have the same number of elements.`);
  util.assert(
      numClasses > 0 && Number.isInteger(numClasses),
      () => `numClasses is required to be a positive integer, but got ` +
          `${numClasses}`);
  // TODO(cais): In the future, if oneHot supports tensors inputs for
  //   `numClasses`, `confusionMatrix` can make `numClasses` optional.

  const oneHotLabels = oneHot(cast($labels, 'int32'), numClasses) as Tensor2D;
  const oneHotPredictions =
      oneHot(cast($predictions, 'int32'), numClasses) as Tensor2D;
  const oneHotLabelsT: Tensor2D = transpose(oneHotLabels);
  return cast(matMul(oneHotLabelsT, oneHotPredictions), 'int32');
}

export const confusionMatrix = op({confusionMatrix_});
