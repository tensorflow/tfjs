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

import {ENGINE} from '../engine';
import {NumericTensor, Tensor} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import {assert, assertShapesMatch} from '../util';

import {op} from './operation';

/**
 * Says whether the targets are in the top K predictions.
 *
 * ```js
 * const predictions = tf.tensor2d([[20, 10, 40, 30], [30, 50, -20, 10]]);
 * const targets = tf.tensor1d([2, 0]);
 * const precision = tf.inTopK(predictions, targets);
 * precision.print();
 * ```
 * @param predictions 2-D or higher `tf.Tensor` with last dimension being
 *     at least `k`.
 * @param targets 1-D or higher `tf.Tensor`.
 * @param k Optional Number of top elements to look at for computing precision,
 *     default to 1.
 */
/** @doc {heading: 'Operations', subheading: 'Evaluation'} */
function inTopK_<T extends Tensor, U extends Tensor>(
    predictions: T|TensorLike, targets: U|TensorLike, k = 1): U {
  const $predictions = convertToTensor(predictions, 'predictions', 'inTopK');
  const $targets = convertToTensor(targets, 'targets', 'inTopK');

  assert(
      $predictions.rank > 1,
      () => 'inTopK() expects the predictions to be of rank 2 or higher, ' +
          `but got ${$predictions.rank}`);
  assert(
      $predictions.rank - 1 === $targets.rank,
      () => `predictions' rank should be 1 larger than ` +
          `targets' rank, but got predictions' rank ` +
          `${$predictions.rank} and targets' rank ${$targets.rank}`);
  assertShapesMatch(
      $predictions.shape.slice(0, $predictions.shape.length - 1),
      $targets.shape,
      `predictions's shape should be align with the targets' shape, ` +
          'except the last dimension.');
  const lastDim = $predictions.shape[$predictions.shape.length - 1];
  assert(
      k > 0 && k <= lastDim,
      () => `'k' passed to inTopK() must be > 0 && <= the predictions' last ` +
          `dimension (${lastDim}), but got ${k}`);

  const precision = ENGINE.runKernel(
      b =>
          b.inTopK($predictions as NumericTensor, $targets as NumericTensor, k),
      {$predictions, $targets});

  return precision as U;
}

export const inTopK = op({inTopK_});
