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
import * as tfc from '@tensorflow/tfjs-core';
import {Tensor, Tensor1D, tidy, util} from '@tensorflow/tfjs-core';

import {epsilon} from './backend/common';
import * as K from './backend/tfjs_backend';
import {ValueError} from './errors';
import {LossOrMetricFn} from './types';

/**
 * Normalizes a tensor wrt the L2 norm alongside the specified axis.
 * @param x
 * @param axis Axis along which to perform normalization.
 */
export function l2Normalize(x: Tensor, axis?: number): Tensor {
  return tidy(() => {
    if (x.dtype !== 'float32') {
      x = x.asType('float32');
    }
    const squareSum = tfc.sum(K.square(x), axis, true);
    const epsilonTensor = tfc.fill(squareSum.shape, epsilon());
    const norm = tfc.sqrt(tfc.maximum(squareSum, epsilonTensor));
    return tfc.div(x, norm);
  });
}

export function meanSquaredError(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => tfc.mean(K.square(tfc.sub(yPred, yTrue)), -1));
}

export function meanAbsoluteError(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => tfc.mean(tfc.abs(tfc.sub(yPred, yTrue)), -1));
}

export function meanAbsolutePercentageError(
    yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    const diff = tfc.sub(yTrue, yPred);
    const clippedTrue =
        tfc.clipByValue(tfc.abs(yTrue), epsilon(), Number.MAX_VALUE);
    const absResult = tfc.abs(tfc.div(diff, clippedTrue));
    return tfc.mul(100, tfc.mean(absResult, -1));
  });
}

export function meanSquaredLogarithmicError(
    yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    const clippedPred = tfc.clipByValue(yPred, epsilon(), Number.MAX_VALUE);
    const firstLog = tfc.log(tfc.add(1, clippedPred));

    const clippedTrue = tfc.clipByValue(yTrue, epsilon(), Number.MAX_VALUE);
    const secondLog = tfc.log(tfc.add(1, clippedTrue));

    return tfc.mean(K.square(tfc.sub(firstLog, secondLog)), -1);
  });
}

export function squaredHinge(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    const maxResult = tfc.maximum(0, tfc.sub(1, tfc.mul(yTrue, yPred)));
    return tfc.mean(K.square(maxResult), -1);
  });
}

export function hinge(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    const maxResult = tfc.maximum(0, tfc.sub(1, tfc.mul(yTrue, yPred)));
    return tfc.mean(maxResult, -1);
  });
}

export function categoricalHinge(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    const pos = tfc.sum(tfc.mul(yTrue, yPred), -1);
    const neg = tfc.max(tfc.mul(tfc.sub(1, yTrue), yPred), -1);
    return tfc.maximum(0, tfc.add(1, tfc.sub(neg, pos)));
  });
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
  return tidy(() => {
    const log2 = Math.log(2);
    const predictionDiff = tfc.sub(yPred, yTrue);
    const logcoshResult = tfc.sub(
        tfc.add(predictionDiff, tfc.softplus(tfc.mul(-2, predictionDiff))),
        log2);
    return tfc.mean(logcoshResult, -1);
  });
}

export function categoricalCrossentropy(
    target: Tensor, output: Tensor, fromLogits = false): Tensor {
  return tidy(() => {
    if (fromLogits) {
      output = tfc.softmax(output);
    } else {
      // scale preds so that the class probabilities of each sample sum to 1.
      const outputSum = tfc.sum(output, output.shape.length - 1, true);
      output = tfc.div(output, outputSum);
    }
    output = tfc.clipByValue(output, epsilon(), 1 - epsilon());
    return tfc.neg(tfc.sum(
        tfc.mul(target.toFloat(), tfc.log(output)), output.shape.length - 1));
  });
}

/**
 * Categorical crossentropy with integer targets.
 *
 * @param target An integer tensor.
 * @param output A tensor resulting from a softmax (unless `fromLogits` is
 *  `true`, in which case `output` is expected to be the logits).
 * @param fromLogits Boolean, whether `output` is the result of a softmax, or is
 *   a tensor of logits.
 */
export function sparseCategoricalCrossentropy(
    target: Tensor, output: Tensor): Tensor {
  return tidy(() => {
    const flatTarget = tfc.floor(K.flatten(target)).toInt() as Tensor1D;
    output = tfc.clipByValue(output, epsilon(), 1 - epsilon());
    const outputShape = output.shape;
    const oneHotTarget =
        tfc.oneHot(flatTarget, outputShape[outputShape.length - 1])
            .reshape(outputShape);
    const fromLogits = false;
    return categoricalCrossentropy(oneHotTarget, output, fromLogits);
  });
}

/**
 * From TensorFlow's implementation in nn_impl.py:
 *
 * For brevity, let `x = logits`, `z = labels`.  The logistic loss is
 *      z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
 *    = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
 *    = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
 *    = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
 *    = (1 - z) * x + log(1 + exp(-x))
 *    = x - x * z + log(1 + exp(-x))
 * For x < 0, to avoid overflow in exp(-x), we reformulate the above
 *      x - x * z + log(1 + exp(-x))
 *    = log(exp(x)) - x * z + log(1 + exp(-x))
 *    = - x * z + log(1 + exp(x))
 * Hence, to ensure stability and avoid overflow, the implementation uses this
 * equivalent formulation
 *    max(x, 0) - x * z + log(1 + exp(-abs(x)))
 *
 * @param labels The labels.
 * @param logits The logits.
 */
export function sigmoidCrossEntropyWithLogits(
    labels: Tensor, logits: Tensor): Tensor {
  if (!util.arraysEqual(labels.shape, logits.shape)) {
    throw new ValueError(
        `logits and labels must have the same shape, but got shapes ` +
        `${JSON.stringify(labels.shape)} and ${JSON.stringify(logits.shape)}`);
  }
  return tidy(() => {
    // The logistic loss formula from above is
    //   x - x * z + log(1 + exp(-x))
    // For x < 0, a more numerically stable formula is
    //   -x * z + log(1 + exp(x))
    // Note that these two expressions can be combined into the following:
    //   max(x, 0) - x * z + log(1 + exp(-abs(x)))
    const reluLogits = logits.relu();
    const negAbsLogits = logits.abs().neg();
    return reluLogits.sub(logits.mul(labels)).add(negAbsLogits.exp().log1p());
  });
}

export function binaryCrossentropy(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    let y: Tensor;
    y = tfc.clipByValue(yPred, epsilon(), 1 - epsilon());
    y = tfc.log(tfc.div(y, tfc.sub(1, y)));
    return tfc.mean(sigmoidCrossEntropyWithLogits(yTrue, y), -1);
  });
}

export function kullbackLeiblerDivergence(
    yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    const clippedTrue = tfc.clipByValue(yTrue, epsilon(), 1);
    const clippedPred = tfc.clipByValue(yPred, epsilon(), 1);
    return tfc.sum(
        tfc.mul(yTrue, tfc.log(tfc.div(clippedTrue, clippedPred))), -1);
  });
}

export function poisson(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    const logPred = tfc.log(tfc.add(epsilon(), yPred));
    return tfc.mean(tfc.sub(yPred, tfc.mul(yTrue, logPred)), -1);
  });
}

export function cosineProximity(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    const trueNormalized = l2Normalize(yTrue, -1);
    const predNormalized = l2Normalize(yPred, -1);
    const trueXPred = tfc.mul(trueNormalized, predNormalized);
    return tfc.neg(tfc.sum(trueXPred, -1));
  });
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

export const lossesMap: {[functionName: string]: LossOrMetricFn} = {
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

// Porting note: This diverges from the PyKeras implementation and may need to
// change based on (de)serialization requirements.
export function get(identifierOrFn: string|LossOrMetricFn): LossOrMetricFn {
  if (typeof identifierOrFn === 'string') {
    if (identifierOrFn in lossesMap) {
      return lossesMap[identifierOrFn];
    }
    let errMsg = `Unknown loss ${identifierOrFn}`;
    if (identifierOrFn.toLowerCase().includes('softmaxcrossentropy')) {
      errMsg = `Unknown loss ${identifierOrFn}. ` +
          'Use "categoricalCrossentropy" as the string name for ' +
          'tf.losses.softmaxCrossEntropy';
    }
    throw new ValueError(errMsg);
  } else {
    return identifierOrFn;
  }
}
