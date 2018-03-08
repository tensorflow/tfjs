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
import {onesLike, Tensor} from 'deeplearn';
import * as K from './backend/deeplearnjs_backend';
import {NotImplementedError, ValueError} from './errors';
import {cosineProximity, meanAbsoluteError, meanAbsolutePercentageError, meanSquaredError} from './losses';
import {LossOrMetricFn} from './types';

// tslint:enable:max-line-length

export function binaryAccuracy(yTrue: Tensor, yPred: Tensor): Tensor {
  // TODO(cais): Maybe avoid creating a new Scalar on every invocation.
  const threshold = K.scalarTimesArray(K.getScalar(0.5), onesLike(yPred));
  const yPredThresholded = K.cast(K.greater(yPred, threshold), yTrue.dtype);
  return K.mean(K.equal(yTrue, yPredThresholded), -1);
}

export function categoricalAccuracy(yTrue: Tensor, yPred: Tensor): Tensor {
  return K.cast(K.equal(K.argmax(yTrue, -1), K.argmax(yPred, -1)), 'float32');
}

export function binaryCrossentropy(yTrue: Tensor, yPred: Tensor): Tensor {
  throw new NotImplementedError();
}

export function categoricalCrossentropy(yTrue: Tensor, yPred: Tensor): Tensor {
  throw new NotImplementedError();
}

export function sparseCategoricalAccuracy(
    yTrue: Tensor, yPred: Tensor): Tensor {
  throw new NotImplementedError();
}

export function sparseCategoricalCrossentropy(
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
export const cosine = cosineProximity;

// TODO(cais, nielsene): Add serialize().

export function get(identifier: string|LossOrMetricFn): LossOrMetricFn {
  const metricsMap: {[functionName: string]: LossOrMetricFn} = {
    binaryAccuracy,
    categoricalAccuracy,
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
