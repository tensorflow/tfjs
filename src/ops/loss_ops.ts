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

import {doc} from '../doc';
import {Tensor} from '../tensor';
import * as util from '../util';

import {operation} from './operation';
import * as ops from './ops';

export enum Reduction {
  NONE,
  MEAN,
  SUM,
  SUM_BY_NONZERO_WEIGHTS
}

export class LossOps {
  /**
   * Computes the weighted loss between two tensors.
   *
   * @param losses Tensor of shape `[batch_size, d1, ... dN]`.
   * @param weights Tensor whose rank is either 0, or the same rank as
   *    `losses`, and must be broadcastable to `losses` (i.e., all
   *    dimensions must be either `1`, or the same as the corresponding
   *    `losses` dimension).
   */
  @doc({heading: 'Training', subheading: 'Losses', namespace: 'losses'})
  @operation
  static computeWeightedLoss<T extends Tensor, O extends Tensor>(
      losses: T, weights?: Tensor,
      reduction = Reduction.SUM_BY_NONZERO_WEIGHTS): O {
    util.assertArgumentsAreTensors({losses}, 'computeWeightedLoss');
    if (weights != null) {
      util.assertArgumentsAreTensors({weights}, 'computeWeightedLoss');
    }

    const weightedLoss = (weights == null) ? losses : losses.mul(weights);

    if (reduction === Reduction.NONE) {
      return weightedLoss as O;
    }
    if (reduction === Reduction.SUM) {
      return weightedLoss.sum();
    }
    if (reduction === Reduction.MEAN) {
      return (weights == null) ? weightedLoss.mean() :
                                 weightedLoss.sum().div(weights.sum());
    }
    if (reduction === Reduction.SUM_BY_NONZERO_WEIGHTS) {
      if (weights == null) {
        return weightedLoss.sum().div(ops.scalar(losses.size));
      } else {
        const numNonZeros = weights.notEqual(ops.scalar(0)).sum().toFloat();
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
  @doc({heading: 'Training', subheading: 'Losses', namespace: 'losses'})
  @operation
  static absoluteDifference<T extends Tensor, O extends Tensor>(
      labels: T, predictions: T, weights?: Tensor,
      reduction = Reduction.SUM_BY_NONZERO_WEIGHTS): O {
    util.assertArgumentsAreTensors({labels, predictions}, 'absoluteDifference');
    if (weights != null) {
      util.assertArgumentsAreTensors({weights}, 'absoluteDifference');
    }
    util.assertShapesMatch(
        labels.shape, predictions.shape, 'Error in absoluteDifference: ');

    const losses = labels.sub(predictions).abs();
    return LossOps.computeWeightedLoss(losses, weights, reduction);
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
  @doc({heading: 'Training', subheading: 'Losses', namespace: 'losses'})
  @operation
  static meanSquaredError<T extends Tensor, O extends Tensor>(
      labels: T, predictions: T, weights?: Tensor,
      reduction = Reduction.SUM_BY_NONZERO_WEIGHTS): O {
    util.assertArgumentsAreTensors({labels, predictions}, 'meanSquaredError');
    if (weights != null) {
      util.assertArgumentsAreTensors({weights}, 'meanSquaredError');
    }
    util.assertShapesMatch(
        labels.shape, predictions.shape, 'Error in meanSquaredError: ');

    const losses = labels.squaredDifference(predictions);
    return LossOps.computeWeightedLoss(losses, weights, reduction);
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
  @doc({heading: 'Training', subheading: 'Losses', namespace: 'losses'})
  @operation
  static cosineDistance<T extends Tensor, O extends Tensor>(
      labels: T, predictions: T, axis: number, weights?: Tensor,
      reduction = Reduction.SUM_BY_NONZERO_WEIGHTS): O {
    util.assertArgumentsAreTensors({labels, predictions}, 'cosineDistance');
    if (weights != null) {
      util.assertArgumentsAreTensors({weights}, 'cosineDistance');
    }
    util.assertShapesMatch(
        labels.shape, predictions.shape, 'Error in cosineDistance: ');

    const one = ops.scalar(1);
    const losses = one.sub(labels.mul(predictions).sum(axis, true));
    return LossOps.computeWeightedLoss(losses, weights, reduction);
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
  @doc({heading: 'Training', subheading: 'Losses', namespace: 'losses'})
  @operation
  static hingeLoss<T extends Tensor, O extends Tensor>(
      labels: T, predictions: T, weights?: Tensor,
      reduction = Reduction.SUM_BY_NONZERO_WEIGHTS): O {
    util.assertArgumentsAreTensors({labels, predictions}, 'hingeLoss');
    if (weights != null) {
      util.assertArgumentsAreTensors({weights}, 'hingeLoss');
    }
    util.assertShapesMatch(
        labels.shape, predictions.shape, 'Error in hingeLoss: ');

    const one = ops.scalar(1);
    // Convert binary labels to (-1, 1)
    labels = ops.scalar(2).mul(labels).sub(one);
    const losses = one.sub(labels.mul(predictions)).relu();
    return LossOps.computeWeightedLoss(losses, weights, reduction);
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
  @doc({heading: 'Training', subheading: 'Losses', namespace: 'losses'})
  @operation
  static logLoss<T extends Tensor, O extends Tensor>(
      labels: T, predictions: T, weights?: Tensor, epsilon = 1e-7,
      reduction = Reduction.SUM_BY_NONZERO_WEIGHTS): O {
    util.assertArgumentsAreTensors({labels, predictions}, 'logLoss');
    if (weights != null) {
      util.assertArgumentsAreTensors({weights}, 'logLoss');
    }
    util.assertShapesMatch(
        labels.shape, predictions.shape, 'Error in logLoss: ');

    const one = ops.scalar(1);
    const epsilonScalar = ops.scalar(epsilon);
    const losses = labels.mul(predictions.add(epsilonScalar).log())
                       .neg()
                       .sub(one.sub(labels).mul(
                           one.sub(predictions).add(epsilonScalar).log()));
    return LossOps.computeWeightedLoss(losses, weights, reduction);
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
  @doc({heading: 'Training', subheading: 'Losses', namespace: 'losses'})
  @operation
  static huberLoss<T extends Tensor, O extends Tensor>(
      labels: T, predictions: T, weights?: Tensor, delta = 1.0,
      reduction = Reduction.SUM_BY_NONZERO_WEIGHTS): O {
    util.assertArgumentsAreTensors({labels, predictions}, 'huberLoss');
    if (weights != null) {
      util.assertArgumentsAreTensors({weights}, 'huberLoss');
    }
    util.assertShapesMatch(
        labels.shape, predictions.shape, 'Error in huberLoss: ');

    const deltaScalar = ops.scalar(delta);
    const error = predictions.sub(labels).abs();
    const quadratic = ops.minimum(error, deltaScalar);
    const linear = error.sub(quadratic);

    const losses =
        ops.scalar(0.5).mul(quadratic.square()).add(deltaScalar.mul(linear));
    return LossOps.computeWeightedLoss(losses, weights, reduction);
  }
}
