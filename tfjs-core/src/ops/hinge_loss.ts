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
import {relu} from './relu';
import {scalar} from './scalar';
import {sub} from './sub';

/**
 * Computes the Hinge loss between two tensors.
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
 */
/** @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'} */
function hingeLoss_<T extends Tensor, O extends Tensor>(
    labels: T|TensorLike, predictions: T|TensorLike,
    weights?: Tensor|TensorLike,
    reduction = Reduction.SUM_BY_NONZERO_WEIGHTS): O {
  let $labels = convertToTensor(labels, 'labels', 'hingeLoss');
  const $predictions = convertToTensor(predictions, 'predictions', 'hingeLoss');
  let $weights: Tensor = null;
  if (weights != null) {
    $weights = convertToTensor(weights, 'weights', 'hingeLoss');
  }
  assertShapesMatch($labels.shape, $predictions.shape, 'Error in hingeLoss: ');

  const one = scalar(1);
  // Convert binary labels to (-1, 1)
  $labels = sub(mul(scalar(2), $labels), one);
  const losses = relu(sub(one, mul($labels, $predictions)));
  return computeWeightedLoss(losses, $weights, reduction);
}
export const hingeLoss = op({hingeLoss_});
