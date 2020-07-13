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
import {Tensor} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import {assertShapesMatch} from '../util';

import {computeWeightedLoss} from './compute_weighted_loss';
import {Reduction} from './loss_ops_utils';
import {mul} from './mul';
import {op} from './operation';
import {scalar} from './scalar';
import {sub} from './sub';
import {sum} from './sum';

/**
 * Computes the cosine distance loss between two tensors.
 *
 * @param labels The ground truth output tensor, same dimensions as
 *    'predictions'.
 * @param predictions The predicted outputs.
 * @param axis The dimension along which the cosine distance is computed.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
 *    must be either `1`, or the same as the corresponding `losses`
 *    dimension).
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`
 */
/** @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'} */
function cosineDistance_<T extends Tensor, O extends Tensor>(
    labels: T|TensorLike, predictions: T|TensorLike, axis: number,
    weights?: Tensor|TensorLike,
    reduction = Reduction.SUM_BY_NONZERO_WEIGHTS): O {
  const $labels = convertToTensor(labels, 'labels', 'cosineDistance');
  const $predictions =
      convertToTensor(predictions, 'predictions', 'cosineDistance');
  let $weights: Tensor = null;
  if (weights != null) {
    $weights = convertToTensor(weights, 'weights', 'cosineDistance');
  }
  assertShapesMatch(
      $labels.shape, $predictions.shape, 'Error in cosineDistance: ');

  const one = scalar(1);
  const losses = sub(one, sum(mul($labels, $predictions), axis, true));
  return computeWeightedLoss(losses, $weights, reduction);
}
export const cosineDistance = op({cosineDistance_});
