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
import {customGrad} from '../gradients';
import {Tensor} from '../tensor';
import {GradSaveFunc} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import {assertShapesMatch} from '../util';

import {add} from './add';
import {expandShapeToKeepDim} from './axis_util';
import {cast} from './cast';
import {computeWeightedLoss} from './compute_weighted_loss';
import {div} from './div';
import {exp} from './exp';
import {logSumExp} from './log_sum_exp';
import {Reduction} from './loss_ops_utils';
import {mul} from './mul';
import {neg} from './neg';
import {op} from './operation';
import {reshape} from './reshape';
import {scalar} from './scalar';
import {sub} from './sub';
import {sum} from './sum';

/**
 * Computes softmax cross entropy between logits and labels.
 *
 * Measures the probability error in discrete classification tasks in which
 * the classes are mutually exclusive (each entry is in exactly one class).
 * For example, each CIFAR-10 image is labeled with one and only one label: an
 * image can be a dog or a truck, but not both.
 *
 * `NOTE`: While the classes are mutually exclusive, their probabilities need
 * not be. All that is required is that each row of labels is a valid
 * probability distribution. If they are not, the computation of the gradient
 * will be incorrect.
 *
 * `WARNING`: This op expects unscaled logits, since it performs a softmax on
 * logits internally for efficiency. Do not call this op with the output of
 * softmax, as it will produce incorrect results.
 *
 * logits and labels must have the same shape, e.g. [batch_size, num_classes]
 * and the same dtype.
 * @param labels The labels array.
 * @param logits The logits array.
 * @param dim The dimension softmax would be performed on. Defaults to `-1`
 *     which indicates the last dimension.
 */
function softmaxCrossEntropyWithLogits_<T extends Tensor, O extends Tensor>(
    labels: T, logits: T, dim = -1): O {
  if (dim === -1) {
    dim = logits.rank - 1;
  }

  if (dim !== logits.rank - 1) {
    throw Error(
        `Softmax cross entropy along a non-last dimension is not yet ` +
        `supported. Labels / logits was rank ${logits.rank} ` +
        `and dim was ${dim}`);
  }
  // Use a custom gradient for numerical stability.
  const customOp =
      customGrad((labels: Tensor, logits: Tensor, save: GradSaveFunc) => {
        // Reference:
        //   1. http://cs231n.github.io/linear-classify/#softmax
        //   2. https://blog.feedly.com/tricks-of-the-trade-logsumexp/
        const keepDims = true;
        const lse = logSumExp(logits, [dim], keepDims);
        const logResult = sub(cast(logits, 'float32'), lse);
        save([labels, logResult]);

        const costVector = neg(mul(logResult, labels));
        const value: O = sum(costVector, [dim]);

        const gradFunc = (dy: O, saved: Tensor[]) => {
          const [labels, logResult] = saved;
          const dyShape = expandShapeToKeepDim(dy.shape, [dim]);
          return [
            mul(reshape(dy, dyShape),
                sub(cast(labels, 'float32'), exp(logResult))),
            mul(reshape(dy, dyShape),
                sub(exp(logResult), cast(labels, 'float32'))),
          ];
        };
        return {value, gradFunc};
      });

  return customOp(labels, logits);
}

/**
 * Computes the softmax cross entropy loss between two tensors.
 *
 * If labelSmoothing is nonzero, smooth the labels towards 1/2:
 *
 *   newOnehotLabels = onehotLabels * (1 - labelSmoothing)
 *                         + labelSmoothing / numClasses
 *
 * @param onehotLabels One hot encoded labels
 *    [batch_size, num_classes], same dimensions as 'predictions'.
 * @param logits The predicted outputs.
 * @param weights Tensor whose rank is either 0, or 1, and must be
 *    broadcastable to `loss`  of shape [batch_size]
 * @param labelSmoothing If greater than 0, then smooth the labels.
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`
 */
/** @doc { heading: 'Training', subheading: 'Losses', namespace: 'losses' } */
function softmaxCrossEntropy_<T extends Tensor, O extends Tensor>(
    onehotLabels: T|TensorLike, logits: T|TensorLike,
    weights?: Tensor|TensorLike, labelSmoothing = 0,
    reduction = Reduction.SUM_BY_NONZERO_WEIGHTS): O {
  let $onehotLabels =
      convertToTensor(onehotLabels, 'onehotLabels', 'softmaxCrossEntropy');
  const $logits = convertToTensor(logits, 'logits', 'softmaxCrossEntropy');
  let $weights: Tensor = null;

  if (weights != null) {
    $weights = convertToTensor(weights, 'weights', 'softmaxCrossEntropy');
  }

  assertShapesMatch(
      $onehotLabels.shape, $logits.shape, 'Error in softmaxCrossEntropy: ');

  if (labelSmoothing > 0) {
    const labelSmoothingScalar = scalar(labelSmoothing);
    const one = scalar(1);
    const numClasses = scalar($onehotLabels.shape[1]);

    $onehotLabels =
        add(mul($onehotLabels, sub(one, labelSmoothingScalar)),
            div(labelSmoothingScalar, numClasses));
  }

  const losses = softmaxCrossEntropyWithLogits_($onehotLabels, $logits);

  return computeWeightedLoss(losses, $weights, reduction);
}

export const softmaxCrossEntropy = op({softmaxCrossEntropy_});
