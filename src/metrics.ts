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

import * as tfc from '@tensorflow/tfjs-core';
import {Tensor, tidy} from '@tensorflow/tfjs-core';
import * as K from './backend/tfjs_backend';
import {NotImplementedError, ValueError} from './errors';
import {categoricalCrossentropy as categoricalCrossentropyLoss, cosineProximity, meanAbsoluteError, meanAbsolutePercentageError, meanSquaredError, sparseCategoricalCrossentropy as sparseCategoricalCrossentropyLoss} from './losses';
import {binaryCrossentropy as lossBinaryCrossentropy} from './losses';
import {LossOrMetricFn} from './types';

export function binaryAccuracy(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    const threshold = tfc.mul(.5, tfc.onesLike(yPred));
    const yPredThresholded = K.cast(tfc.greater(yPred, threshold), yTrue.dtype);
    return tfc.mean(tfc.equal(yTrue, yPredThresholded), -1);
  });
}

export function categoricalAccuracy(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(
      () => K.cast(
          tfc.equal(tfc.argMax(yTrue, -1), tfc.argMax(yPred, -1)), 'float32'));
}

function truePositives(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    return tfc.logicalAnd(yTrue.equal(1), yPred.equal(1)).sum().cast('float32');
  });
}

function falseNegatives(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    return tfc.logicalAnd(yTrue.equal(1), yPred.equal(0)).sum().cast('float32');
  });
}

function falsePositives(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    return tfc.logicalAnd(yTrue.equal(0), yPred.equal(1)).sum().cast('float32');
  });
}

export function precision(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    const tp = truePositives(yTrue, yPred);
    const fp = falsePositives(yTrue, yPred);

    const denominator = tp.add(fp);

    return tfc.where(tfc.greater(denominator, 0), tp.div(denominator), 0)
        .cast('float32');
  });
}

export function recall(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    const tp = truePositives(yTrue, yPred);
    const fn = falseNegatives(yTrue, yPred);

    const denominator = tp.add(fn);

    return tfc.where(tfc.greater(denominator, 0), tp.div(denominator), 0)
        .cast('float32');
  });
}

export function binaryCrossentropy(yTrue: Tensor, yPred: Tensor): Tensor {
  return lossBinaryCrossentropy(yTrue, yPred);
}

export function sparseCategoricalAccuracy(
    yTrue: Tensor, yPred: Tensor): Tensor {
  if (yTrue.rank === yPred.rank) {
    yTrue = yTrue.squeeze([yTrue.rank - 1]);
  }
  yPred = yPred.argMax(-1);
  if (yPred.dtype !== yTrue.dtype) {
    yPred = yPred.asType(yTrue.dtype);
  }
  return tfc.equal(yTrue, yPred).asType('float32');
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
    precision,
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
