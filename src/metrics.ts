/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Built-in metrics.
 */

// tslint:disable:max-line-length
import * as tfc from '@tensorflow/tfjs-core';
import {Tensor, tidy} from '@tensorflow/tfjs-core';

import * as K from './backend/tfjs_backend';
import {NotImplementedError, ValueError} from './errors';
import {categoricalCrossentropy as categoricalCrossentropyLoss, cosineProximity, meanAbsoluteError, meanAbsolutePercentageError, meanSquaredError, sparseCategoricalCrossentropy as sparseCategoricalCrossentropyLoss} from './losses';
import {binaryCrossentropy as lossBinaryCrossentropy} from './losses';
import {LossOrMetricFn} from './types';
// tslint:enable:max-line-length

/**
 * Binary accuracy metric function.
 *
 * `yTrue` and `yPred` can have 0-1 values. Example:
 * ```js
 * const x = tensor2d([[1, 1, 1, 1], [0, 0, 0, 0]], [2, 4]);
 * const y = tensor2d([[1, 0, 1, 0], [0, 0, 0, 1]], [2, 4]);
 * const accuracy = tfl.metrics.binaryAccuracy(x, y);
 * accuracy.print();
 * ```
 *
 * `yTrue` and `yPred` can also have floating-number values between 0 and 1, in
 * which case the values will be thresholded at 0.5 to yield 0-1 values (i.e.,
 * a value >= 0.5 and <= 1.0 is interpreted as 1.
 * )
 * Example:
 * ```js
 * const x = tensor1d([1, 1, 1, 1, 0, 0, 0, 0]);
 * const y = tensor1d([0.2, 0.4, 0.6, 0.8, 0.2, 0.3, 0.4, 0.7]);
 * const accuracy = tf.metrics.binaryAccuracy(x, y);
 * accuracy.print();
 * ```
 *
 * @param yTrue Binary Tensor of truth.
 * @param yPred Binary Tensor of prediction.
 * @return Accuracy Tensor.
 */
export function binaryAccuracy(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    const threshold = K.scalarTimesArray(K.getScalar(0.5), tfc.onesLike(yPred));
    const yPredThresholded = K.cast(tfc.greater(yPred, threshold), yTrue.dtype);
    return tfc.mean(tfc.equal(yTrue, yPredThresholded), -1);
  });
}

/**
 * Categorical accuracy metric function.
 *
 * Example:
 * ```js
 * const x = tensor2d([[0, 0, 0, 1], [0, 0, 0, 1]]);
 * const y = tensor2d([[0.1, 0.8, 0.05, 0.05], [0.1, 0.05, 0.05, 0.8]]);
 * const accuracy = tf.metrics.categoricalAccuracy(x, y);
 * accuracy.print();
 * ```
 *
 * @param yTrue Binary Tensor of truth: one-hot encoding of categories.
 * @param yPred Binary Tensor of prediction: probabilities or logits for the
 *   same categories as in `yTrue`.
 * @return Accuracy Tensor.
 */
export function categoricalAccuracy(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(
      () => K.cast(
          tfc.equal(tfc.argMax(yTrue, -1), tfc.argMax(yPred, -1)), 'float32'));
}

/**
 * Binary crossentropy metric function.
 *
 * Example:
 * ```js
 * const x = tensor2d([[0], [1], [1], [1]]);
 * const y = tensor2d([[0], [0], [0.5], [1]]);
 * const crossentropy = tf.metrics.binaryCrossentropy(x, y);
 * crossentropy.print();
 * ```
 *
 * @param yTrue Binary Tensor of truth.
 * @param yPred Binary Tensor of prediction, probabilities for the `1` case.
 * @return Accuracy Tensor.
 */
export function binaryCrossentropy(yTrue: Tensor, yPred: Tensor): Tensor {
  return lossBinaryCrossentropy(yTrue, yPred);
}

export function sparseCategoricalAccuracy(
    yTrue: Tensor, yPred: Tensor): Tensor {
  throw new NotImplementedError();
}

export function topKCategoricalAccuracy(yTrue: Tensor, yPred: Tensor): Tensor {
  throw new NotImplementedError();
}

export function sparseTopKCategoricalAccuracy(
    yTrue: Tensor, yPred: Tensor): Tensor {
  throw new NotImplementedError();
}

// Aliases.
export const mse = meanSquaredError;
export const MSE = meanSquaredError;
export const mae = meanAbsoluteError;
export const MAE = meanAbsoluteError;
export const mape = meanAbsolutePercentageError;
export const MAPE = meanAbsolutePercentageError;
export const categoricalCrossentropy = categoricalCrossentropyLoss;
export const cosine = cosineProximity;
export const sparseCategoricalCrossentropy = sparseCategoricalCrossentropyLoss;

// TODO(cais, nielsene): Add serialize().

export function get(identifier: string|LossOrMetricFn): LossOrMetricFn {
  const metricsMap: {[functionName: string]: LossOrMetricFn} = {
    binaryAccuracy,
    categoricalAccuracy,
    categoricalCrossentropy,
    sparseCategoricalCrossentropy,
    mse,
    MSE,
    mae,
    MAE,
    mape,
    MAPE,
    cosine,
  };
  if (typeof identifier === 'string' && identifier in metricsMap) {
    return metricsMap[identifier];
  } else if (typeof identifier !== 'string' && identifier != null) {
    return identifier;
  } else {
    throw new ValueError(`Unknown metric ${identifier}`);
  }
}
