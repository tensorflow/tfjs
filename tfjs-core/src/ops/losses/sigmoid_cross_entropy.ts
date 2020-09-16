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
import {abs} from '../abs';
import {add} from '../add';
import {exp} from '../exp';
import {log1p} from '../log1p';
import {Reduction} from '../loss_ops_utils';
import {mul} from '../mul';
import {neg} from '../neg';
import {op} from '../operation';
import {relu} from '../relu';
import {scalar} from '../scalar';
import {sub} from '../sub';

import {computeWeightedLoss} from './compute_weighted_loss';

function sigmoidCrossEntropyWithLogits_<T extends Tensor, O extends Tensor>(
    labels: T|TensorLike, logits: T|TensorLike): O {
  const $labels =
      convertToTensor(labels, 'labels', 'sigmoidCrossEntropyWithLogits');
  const $logits =
      convertToTensor(logits, 'logits', 'sigmoidCrossEntropyWithLogits');
  assertShapesMatch(
      $labels.shape, $logits.shape, 'Error in sigmoidCrossEntropyWithLogits: ');

  /**
   * Implementation Details:
   *
   * For brevity, let `x = logits`, `z = labels`.  The logistic loss is
   *     z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
   *   = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
   *   = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
   *   = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
   *   = (1 - z) * x + log(1 + exp(-x))
   *   = x - x * z + log(1 + exp(-x))
   *
   *   For x < 0, to avoid overflow in exp(-x), we reformulate the above
   *     x - x * z + log(1 + exp(-x))
   *   = log(exp(x)) - x * z + log(1 + exp(-x))
   *   = - x * z + log(1 + exp(x))
   *
   * Hence, to ensure stability and avoid overflow, the implementation uses
   * this equivalent formulation:
   *     max(x, 0) - x * z + log(1 + exp(-abs(x)))
   */
  const maxOutput = relu($logits);
  const outputXTarget = mul($logits, $labels);
  const sigmoidOutput = log1p(exp(neg(abs($logits))));

  return add(sub(maxOutput, outputXTarget), sigmoidOutput);
}

/**
 * Computes the sigmoid cross entropy loss between two tensors.
 *
 * If labelSmoothing is nonzero, smooth the labels towards 1/2:
 *
 *   newMulticlassLabels = multiclassLabels * (1 - labelSmoothing)
 *                         + 0.5 * labelSmoothing
 *
 * @param multiClassLabels The ground truth output tensor of shape
 * [batch_size, num_classes], same dimensions as 'predictions'.
 * @param logits The predicted outputs.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
 *    must be either `1`, or the same as the corresponding `losses`
 *    dimension).
 * @param labelSmoothing If greater than 0, then smooth the labels.
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`
 *
 * @doc { heading: 'Training', subheading: 'Losses', namespace: 'losses' }
 */
function sigmoidCrossEntropy_<T extends Tensor, O extends Tensor>(
    multiClassLabels: T|TensorLike, logits: T|TensorLike,
    weights?: Tensor|TensorLike, labelSmoothing = 0,
    reduction = Reduction.SUM_BY_NONZERO_WEIGHTS): O {
  let $multiClassLabels = convertToTensor(
      multiClassLabels, 'multiClassLabels', 'sigmoidCrossEntropy');
  const $logits = convertToTensor(logits, 'logits', 'sigmoidCrossEntropy');
  let $weights: Tensor = null;
  if (weights != null) {
    $weights = convertToTensor(weights, 'weights', 'sigmoidCrossEntropy');
  }
  assertShapesMatch(
      $multiClassLabels.shape, $logits.shape, 'Error in sigmoidCrossEntropy: ');

  if (labelSmoothing > 0) {
    const labelSmoothingScalar = scalar(labelSmoothing);
    const one = scalar(1);
    const half = scalar(0.5);

    $multiClassLabels =
        add(mul($multiClassLabels, sub(one, labelSmoothingScalar)),
            mul(half, labelSmoothingScalar));
  }
  const losses = sigmoidCrossEntropyWithLogits_($multiClassLabels, $logits);

  return computeWeightedLoss(losses, $weights, reduction);
}

export const sigmoidCrossEntropy = op({sigmoidCrossEntropy_});
