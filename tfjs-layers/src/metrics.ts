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
import {lossesMap} from './losses';
import {LossOrMetricFn} from './types';
import * as util from './utils/generic_utils';

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

export const metricsMap: {[functionName: string]: LossOrMetricFn} = {
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
  cosine
};

export function get(identifier: string|LossOrMetricFn): LossOrMetricFn {
  if (typeof identifier === 'string' && identifier in metricsMap) {
    return metricsMap[identifier];
  } else if (typeof identifier !== 'string' && identifier != null) {
    return identifier;
  } else {
    throw new ValueError(`Unknown metric ${identifier}`);
  }
}

/**
 * Get the shortcut function name.
 *
 * If the fn name is a string,
 *   directly return the string name.
 * If the function is included in metricsMap or lossesMap,
 *   return key of the map.
 *   - If the function relative to multiple keys,
 *     return the first found key as the function name.
 *   - If the function exists in both lossesMap and metricsMap,
 *     search lossesMap first.
 * If the function is not included in metricsMap or lossesMap,
 *   return the function name.
 *
 * @param fn loss function, metric function, or short cut name.
 * @returns Loss or Metric name in string.
 */
export function getLossOrMetricName(fn: string|LossOrMetricFn): string {
  util.assert(fn !== null, `Unknown LossOrMetricFn ${fn}`);
  if (typeof fn === 'string') {
    return fn;
  } else {
    let fnName;
    for (const key of Object.keys(lossesMap)) {
      if (lossesMap[key] === fn) {
        fnName = key;
        break;
      }
    }
    if (fnName !== undefined) {
      return fnName;
    }
    for (const key of Object.keys(metricsMap)) {
      if (metricsMap[key] === fn) {
        fnName = key;
        break;
      }
    }
    if (fnName !== undefined) {
      return fnName;
    }
    return (fn as Function).name;
  }
}
