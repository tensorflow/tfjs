/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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
import {expandShapeToKeepDim} from './axis_util';
import {minimum} from './minimum';
import {op} from './operation';
import {ones, scalar} from './tensor_ops';

export enum Reduction {
  NONE,
  MEAN,
  SUM,
  SUM_BY_NONZERO_WEIGHTS
}

/**
 * Computes the weighted loss between two tensors.
 *
 * @param losses Tensor of shape `[batch_size, d1, ... dN]`.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `losses`, and must be broadcastable to `losses` (i.e., all
 *    dimensions must be either `1`, or the same as the corresponding
 *    `losses` dimension).
 */
/** @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'} */
function computeWeightedLoss_<T extends Tensor, O extends Tensor>(
    losses: T|TensorLike, weights?: Tensor|TensorLike,
    reduction = Reduction.SUM_BY_NONZERO_WEIGHTS): O {
  const $losses = convertToTensor(losses, 'losses', 'computeWeightedLoss');
  let $weights: Tensor = null;
  if (weights != null) {
    $weights = convertToTensor(weights, 'weights', 'computeWeightedLoss');
  }

  const weightedLoss = ($weights == null) ? $losses : $losses.mul($weights);

  if (reduction === Reduction.NONE) {
    return weightedLoss as O;
  }
  if (reduction === Reduction.SUM) {
    return weightedLoss.sum();
  }
  if (reduction === Reduction.MEAN) {
    if ($weights == null) {
      return weightedLoss.mean();
    } else {
      const broadcastFactor = $losses.size / $weights.size;
      const result = weightedLoss.sum().div($weights.sum());
      return broadcastFactor > 1 ? result.div(scalar(broadcastFactor)) :
                                   result as O;
    }
  }
  if (reduction === Reduction.SUM_BY_NONZERO_WEIGHTS) {
    if ($weights == null) {
      return weightedLoss.sum().div(scalar($losses.size));
    } else {
      const broadcastedWeights = $weights.mul(ones($losses.shape));

      const numNonZeros =
          broadcastedWeights.notEqual(scalar(0)).sum().toFloat();
      return weightedLoss.sum().div(numNonZeros);
    }
  }

  throw Error(`Unknown reduction: ${reduction}`);
}

/**
 * Computes the absolute difference loss between two tensors.
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
function absoluteDifference_<T extends Tensor, O extends Tensor>(
    labels: T|TensorLike, predictions: T|TensorLike,
    weights?: Tensor|TensorLike,
    reduction = Reduction.SUM_BY_NONZERO_WEIGHTS): O {
  const $labels = convertToTensor(labels, 'labels', 'absoluteDifference');
  const $predictions =
      convertToTensor(predictions, 'predictions', 'absoluteDifference');
  let $weights: Tensor = null;
  if (weights != null) {
    $weights = convertToTensor(weights, 'weights', 'absoluteDifference');
  }
  assertShapesMatch(
      $labels.shape, $predictions.shape, 'Error in absoluteDifference: ');

  const losses = $labels.sub($predictions).abs();
  return computeWeightedLoss(losses, $weights, reduction);
}

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
 */
/** @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'} */
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

  const losses = $labels.squaredDifference($predictions);
  return computeWeightedLoss(losses, $weights, reduction);
}

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
  const losses = one.sub($labels.mul($predictions).sum(axis, true));
  return computeWeightedLoss(losses, $weights, reduction);
}

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
  $labels = scalar(2).mul($labels).sub(one);
  const losses = one.sub($labels.mul($predictions)).relu();
  return computeWeightedLoss(losses, $weights, reduction);
}

/**
 * Computes the log loss between two tensors.
 *
 * @param labels The ground truth output tensor, same dimensions as
 *    'predictions'.
 * @param predictions The predicted outputs.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
 *    must be either `1`, or the same as the corresponding `losses`
 *    dimension).
 * @param epsilon A small increment to avoid taking log of zero
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`
 */
/** @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'} */
function logLoss_<T extends Tensor, O extends Tensor>(
    labels: T|TensorLike, predictions: T|TensorLike,
    weights?: Tensor|TensorLike, epsilon = 1e-7,
    reduction = Reduction.SUM_BY_NONZERO_WEIGHTS): O {
  const $labels = convertToTensor(labels, 'labels', 'logLoss');
  const $predictions = convertToTensor(predictions, 'predictions', 'logLoss');
  let $weights: Tensor = null;
  if (weights != null) {
    $weights = convertToTensor(weights, 'weights', 'logLoss');
  }
  assertShapesMatch($labels.shape, $predictions.shape, 'Error in logLoss: ');

  const one = scalar(1);
  const epsilonScalar = scalar(epsilon);
  const losses = $labels.mul($predictions.add(epsilonScalar).log())
                     .neg()
                     .sub(one.sub($labels).mul(
                         one.sub($predictions).add(epsilonScalar).log()));
  return computeWeightedLoss(losses, $weights, reduction);
}

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
  const maxOutput = $logits.relu();
  const outputXTarget = $logits.mul($labels);
  const sigmoidOutput = $logits.abs().neg().exp().log1p();

  return maxOutput.sub(outputXTarget).add(sigmoidOutput);
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
 */
/** @doc { heading: 'Training', subheading: 'Losses', namespace: 'losses' } */
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

    $multiClassLabels = $multiClassLabels.mul(one.sub(labelSmoothingScalar))
                            .add(half.mul(labelSmoothingScalar));
  }
  const losses = sigmoidCrossEntropyWithLogits_($multiClassLabels, $logits);

  return computeWeightedLoss(losses, $weights, reduction);
}

/**
 * Computes the huber loss between two tensors.
 *
 * @param labels The ground truth output tensor, same dimensions as
 *    'predictions'.
 * @param predictions The predicted outputs.
 * @param weights Tensor whose rank is either 0, or the same rank as
 *    `labels`, and must be broadcastable to `labels` (i.e., all dimensions
 *    must be either `1`, or the same as the corresponding `losses`
 *    dimension).
 * @param delta Point where huber loss changes from quadratic to linear.
 * @param reduction Type of reduction to apply to loss. Should be of type
 *    `Reduction`.
 */
/** @doc {heading: 'Training', subheading: 'Losses', namespace: 'losses'} */
function huberLoss_<T extends Tensor, O extends Tensor>(
    labels: T|TensorLike, predictions: T|TensorLike,
    weights?: Tensor|TensorLike, delta = 1.0,
    reduction = Reduction.SUM_BY_NONZERO_WEIGHTS): O {
  const $labels = convertToTensor(labels, 'labels', 'huberLoss');
  const $predictions = convertToTensor(predictions, 'predictions', 'huberLoss');
  let $weights: Tensor = null;
  if (weights != null) {
    $weights = convertToTensor(weights, 'weights', 'huberLoss');
  }
  assertShapesMatch($labels.shape, $predictions.shape, 'Error in huberLoss: ');

  const deltaScalar = scalar(delta);
  const error = $predictions.sub($labels).abs();
  const quadratic = minimum(error, deltaScalar);
  const linear = error.sub(quadratic);

  const losses =
      scalar(0.5).mul(quadratic.square()).add(deltaScalar.mul(linear));
  return computeWeightedLoss(losses, $weights, reduction);
}

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
        const lse = logits.logSumExp([dim], keepDims);
        const logResult = logits.toFloat().sub(lse);
        save([labels, logResult]);

        const costVector = logResult.mul(labels).neg();
        const value: O = costVector.sum([dim]);

        const gradFunc = (dy: O, saved: Tensor[]) => {
          const [labels, logResult] = saved;
          const dyShape = expandShapeToKeepDim(dy.shape, [dim]);
          return [
            dy.reshape(dyShape).mul(labels.toFloat().sub(logResult.exp())),
            dy.reshape(dyShape).mul(logResult.exp().sub(labels.toFloat())),
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

    $onehotLabels = $onehotLabels.mul(one.sub(labelSmoothingScalar))
                        .add(labelSmoothingScalar.div(numClasses));
  }

  const losses = softmaxCrossEntropyWithLogits_($onehotLabels, $logits);

  return computeWeightedLoss(losses, $weights, reduction);
}

export const absoluteDifference = op({absoluteDifference_});
export const computeWeightedLoss = op({computeWeightedLoss_});
export const cosineDistance = op({cosineDistance_});
export const hingeLoss = op({hingeLoss_});
export const huberLoss = op({huberLoss_});
export const logLoss = op({logLoss_});
export const meanSquaredError = op({meanSquaredError_});
export const sigmoidCrossEntropy = op({sigmoidCrossEntropy_});
export const softmaxCrossEntropy = op({softmaxCrossEntropy_});
