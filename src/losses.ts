/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/* Original Source: losses.py */

import {Tensor} from 'deeplearn';

import * as K from './backend/deeplearnjs_backend';
import {ValueError} from './errors';
import {LossOrMetricFn} from './types';

export function meanSquaredError(yTrue: Tensor, yPred: Tensor): Tensor {
  return K.mean(K.square(K.subtract(yPred, yTrue)), -1);
}

export function meanAbsoluteError(yTrue: Tensor, yPred: Tensor): Tensor {
  return K.mean(K.abs(K.subtract(yPred, yTrue)), -1);
}

export function meanAbsolutePercentageError(
    yTrue: Tensor, yPred: Tensor): Tensor {
  const diff = K.subtract(yTrue, yPred);
  const clippedTrue = K.clip(K.abs(yTrue), K.epsilon(), Number.MAX_VALUE);
  const absResult = K.abs(K.divide(diff, clippedTrue));
  return K.scalarTimesArray(K.getScalar(100.0), K.mean(absResult, -1));
}

export function meanSquaredLogarithmicError(
    yTrue: Tensor, yPred: Tensor): Tensor {
  const one = K.getScalar(1.0);

  const clippedPred = K.clip(yPred, K.epsilon(), Number.MAX_VALUE);
  const firstLog = K.log(K.scalarPlusArray(one, clippedPred));

  const clippedTrue = K.clip(yTrue, K.epsilon(), Number.MAX_VALUE);
  const secondLog = K.log(K.scalarPlusArray(one, clippedTrue));

  return K.mean(K.square(K.subtract(firstLog, secondLog)), -1);
}

export function squaredHinge(yTrue: Tensor, yPred: Tensor): Tensor {
  const zeroTensor = K.getScalar(0.0);
  const one = K.getScalar(1.0);
  const maxResult =
      K.maximum(zeroTensor, K.subtract(one, K.multiply(yTrue, yPred)));
  return K.mean(K.square(maxResult), -1);
}

export function hinge(yTrue: Tensor, yPred: Tensor): Tensor {
  const zeroTensor = K.getScalar(0.0);
  const one = K.getScalar(1.0);
  const maxResult =
      K.maximum(zeroTensor, K.subtract(one, K.multiply(yTrue, yPred)));
  return K.mean(maxResult, -1);
}

export function categoricalHinge(yTrue: Tensor, yPred: Tensor): Tensor {
  const zeroTensor = K.getScalar(0.0);
  const one = K.getScalar(1.0);
  const pos = K.sum(K.multiply(yTrue, yPred), -1);
  const neg = K.max(K.multiply(K.subtract(one, yTrue), yPred), -1);
  return K.maximum(zeroTensor, K.scalarPlusArray(one, K.subtract(neg, pos)));
}

/**
 * Logarithm of the hyperbolic cosine of the prediction error.
 *
 * `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
 * to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
 * like the mean squared error, but will not be so strongly affected by the
 * occasional wildly incorrect prediction.
 */
export function logcosh(yTrue: Tensor, yPred: Tensor): Tensor {
  const log2 = K.getScalar(Math.log(2.0));
  const predictionDiff = K.subtract(yPred, yTrue);
  const logcoshResult = K.subtract(
      K.add(
          predictionDiff,
          K.softplus(K.scalarTimesArray(K.getScalar(-2.0), predictionDiff))),
      log2);
  return K.mean(logcoshResult, -1);
}

export function categoricalCrossentropy(yTrue: Tensor, yPred: Tensor): Tensor {
  return K.categoricalCrossentropy(yTrue, yPred);
}

export function sparseCategoricalCrossentropy(
    yTrue: Tensor, yPred: Tensor): Tensor {
  return K.sparseCategoricalCrossentropy(yTrue, yPred);
}

export function binaryCrossentropy(yTrue: Tensor, yPred: Tensor): Tensor {
  return K.mean(K.binaryCrossentropy(yTrue, yPred), -1);
}

export function kullbackLeiblerDivergence(
    yTrue: Tensor, yPred: Tensor): Tensor {
  const clippedTrue = K.clip(yTrue, K.epsilon(), 1);
  const clippedPred = K.clip(yPred, K.epsilon(), 1);
  return K.sum(
      K.multiply(yTrue, K.log(K.divide(clippedTrue, clippedPred))), -1);
}

export function poisson(yTrue: Tensor, yPred: Tensor): Tensor {
  const logPred = K.log(K.scalarPlusArray(K.getScalar(K.epsilon()), yPred));
  return K.mean(K.subtract(yPred, K.multiply(yTrue, logPred)), -1);
}

export function cosineProximity(yTrue: Tensor, yPred: Tensor): Tensor {
  const trueNormalized = K.l2Normalize(yTrue, -1);
  const predNormalized = K.l2Normalize(yPred, -1);
  const trueXPred = K.multiply(trueNormalized, predNormalized);
  return K.neg(K.sum(trueXPred, -1));
}

export const mse = meanSquaredError;
export const MSE = meanSquaredError;
export const mae = meanAbsoluteError;
export const MAE = meanAbsoluteError;
export const mape = meanAbsolutePercentageError;
export const MAPE = meanAbsolutePercentageError;
export const msle = meanSquaredLogarithmicError;
export const MSLE = meanSquaredLogarithmicError;
export const kld = kullbackLeiblerDivergence;
export const KLD = kullbackLeiblerDivergence;
export const cosine = cosineProximity;

// TODO(michaelterry): Add deserialize() function.

// Porting note: This diverges from the PyKeras implementation and may need to
// change based on (de)serialization requirements.
export function get(identifier: string): LossOrMetricFn {
  const lossesMap: {[functionName: string]: LossOrMetricFn} = {
    meanSquaredError,
    meanAbsoluteError,
    meanAbsolutePercentageError,
    meanSquaredLogarithmicError,
    squaredHinge,
    hinge,
    categoricalHinge,
    logcosh,
    categoricalCrossentropy,
    sparseCategoricalCrossentropy,
    binaryCrossentropy,
    kullbackLeiblerDivergence,
    poisson,
    cosineProximity
  };
  if (identifier in lossesMap) {
    return lossesMap[identifier];
  }
  throw new ValueError(`Unknown loss ${identifier}`);
}
