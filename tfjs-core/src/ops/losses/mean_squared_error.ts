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

import {Tensor} from '../../tensor';
import {convertToTensor} from '../../tensor_util_env';
import {TensorLike} from '../../types';
import {assertShapesMatch} from '../../util';
import {Reduction} from '../loss_ops_utils';
import {op} from '../operation';
import {squaredDifference} from '../squared_difference';

import {computeWeightedLoss} from './compute_weighted_loss';

/**
 * Computes the mean squared error between two tensors.
 *
 * @param labels The ground truth output tensor, same dimensions as
 *    'predictions'.
 * @param predictions The predicted outputs.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
 *    must be either `1`, or the same as the corresponding `losses`
 *    dimension).
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`
 *
 * @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'}
 */
function meanSquaredError_<T extends Tensor, O extends Tensor>(
    labels: T|TensorLike, predictions: T|TensorLike,
    weights?: Tensor|TensorLike,
    reduction = Reduction.SUM_BY_NONZERO_WEIGHTS): O {
  const $labels = convertToTensor(labels, 'labels', 'meanSquaredError');
  const $predictions =
      convertToTensor(predictions, 'predictions', 'meanSquaredError');
  let $weights: Tensor = null;
  if (weights != null) {
    $weights = convertToTensor(weights, 'weights', 'meanSquaredError');
  }
  assertShapesMatch(
      $labels.shape, $predictions.shape, 'Error in meanSquaredError: ');

  const losses = squaredDifference($labels, $predictions);
  return computeWeightedLoss(losses, $weights, reduction);
}
export const meanSquaredError = op({meanSquaredError_});
