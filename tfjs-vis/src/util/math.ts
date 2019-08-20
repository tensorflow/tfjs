/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {dispose, maximum, scalar, Tensor, Tensor1D, tidy} from '@tensorflow/tfjs';

import {HistogramStats, TypedArray} from '../types';

import {assert, assertShapesMatch} from './utils';

/**
 * Returns summary statistics for an array of numbers
 *
 * @param input
 */
export function arrayStats(input: number[]): HistogramStats {
  if (!Array.isArray(input)) {
    throw new Error('input must be an array');
  }
  if (input.length === 0) {
    return {
      numVals: 0,
      numNans: 0,
      numZeros: 0,
      max: undefined,
      min: undefined,
    };
  }

  const numVals = input.length;
  let max = -Infinity;
  let min = Infinity;
  let numZeros = 0;
  let numNans = 0;
  let numInfs = 0;

  for (let i = 0; i < numVals; i++) {
    const curr = input[i];
    if (curr > max) {
      max = curr;
    }

    if (curr < min) {
      min = curr;
    }

    if (curr === 0) {
      numZeros += 1;
    }

    if (isNaN(curr)) {
      numNans += 1;
    } else if (!isFinite(curr)) {
      // Make sure NaNs are not double counted as Infs
      numInfs += 1;
    }
  }

  const result = {
    numVals,
    numZeros,
    numNans,
    max,
    min,
    numInfs,
  };

  // Handle all NaN input
  if (result.max === -Infinity) {
    result.max = NaN;
  }
  if (result.min === Infinity) {
    result.min = NaN;
  }

  return result;
}

/**
 * Returns summary statistics for a numeric tensor. *
 *
 * @param input
 */
export async function tensorStats(input: Tensor): Promise<HistogramStats> {
  // TODO. Benchmark this and consider having one of the *stats functions
  // delegate to the other.

  const [min, max, numZeros] = tidy(() => {
    const zero = scalar(0, input.dtype);

    const min = input.min();
    const max = input.max();
    const numZeros = input.equal(zero).sum();

    return [min, max, numZeros];
  });

  return await Promise
      .all([input.data(), min.data(), max.data(), numZeros.data()])
      .then(([tensorVal, minVal, maxVal, numZerosVal]) => {
        // We currently need to count NaNs on CPU.
        const numVals = tensorVal.length;
        let numNans = 0;
        let numInfs = 0;
        for (let i = 0; i < numVals; i++) {
          const curr = tensorVal[i];
          if (isNaN(curr)) {
            numNans += 1;
          } else if (!isFinite(curr)) {
            // Make sure NaNs are not double counted as Infs
            numInfs += 1;
          }
        }

        let trueMin = minVal[0];
        let trueMax = maxVal[0];
        if (numNans === numVals) {
          // on gpu the min and max won't be accurate if all values are NaN
          trueMin = NaN;
          trueMax = NaN;
        }

        const stats = {
          numVals,
          numZeros: numZerosVal[0],
          numNans,
          min: trueMin,
          max: trueMax,
          numInfs,
        };

        return stats;
      });
}

/**
 * Computes a confusion matrix from predictions and labels. Each value in
 * labels and predictions should correspond to some output class. It is assumed
 * that these values go from 0 to numClasses - 1.
 *
 * ```js
 * const labels = tf.tensor1d([1, 2, 4]);
 * const predictions = tf.tensor1d([2, 2, 4]);
 * const result = await tfvis.metrics.confusionMatrix(labels, predictions);
 * console.log(JSON.stringify(result, null, 2))
 * ```
 *
 * @param labels 1D tensor of true values
 * @param predictions 1D tensor of predicted values
 * @param numClasses Number of distinct classes. Optional. If not passed in
 *  numClasses will equal the highest number in either labels or predictions
 *  plus 1
 * @param weights 1d tensor that is the same size as predictions.
 *  If weights is passed in then each prediction contributes its corresponding
 *  weight to the total value of the confusion matrix cell.
 *
 */
/**
 * @doc {heading: 'Metrics', namespace: 'metrics'}
 */
export async function confusionMatrix(
    labels: Tensor1D, predictions: Tensor1D, numClasses?: number,
    weights?: Tensor1D): Promise<number[][]> {
  assert(labels.rank === 1, 'labels must be a 1D tensor');
  assert(predictions.rank === 1, 'predictions must be a 1D tensor');
  assert(
      labels.size === predictions.size,
      'labels and predictions must be the same length');
  if (weights != null) {
    assert(
        weights.size === predictions.size,
        'labels and predictions must be the same length');
  }

  // Cast to int in case the caller didn't
  const labelsInt = labels.cast('int32');
  const predictionsInt = predictions.cast('int32');

  if (numClasses == null) {
    numClasses =
        tidy(() => {
          const max =
              maximum(labelsInt.max(), predictionsInt.max()).cast('int32');
          return max.dataSync()[0] + 1;
        }) as number;
  }

  let weightsPromise: Promise<null|TypedArray> = Promise.resolve(null);
  if (weights != null) {
    weightsPromise = weights.data();
  }

  return Promise.all([labelsInt.data(), predictionsInt.data(), weightsPromise])
      .then(([labelsArray, predsArray, weightsArray]) => {
        const result: number[][] = Array(numClasses).fill(0);
        // Initialize the matrix
        for (let i = 0; i < numClasses!; i++) {
          result[i] = Array(numClasses).fill(0);
        }

        for (let i = 0; i < labelsArray.length; i++) {
          const label = labelsArray[i];
          const pred = predsArray[i];

          if (weightsArray != null) {
            result[label][pred] += weightsArray[i];
          } else {
            result[label][pred] += 1;
          }
        }

        return result;
      });
}

/**
 * Computes how often predictions matches labels
 *
 * ```js
 * const labels = tf.tensor1d([0, 0, 1, 2, 2, 2]);
 * const predictions = tf.tensor1d([0, 0, 0, 2, 1, 1]);
 *
 * const result = await tfvis.metrics.accuracy(labels, predictions);
 * console.log(result)
 * ```
 *
 * @param labels tensor of true values
 * @param predictions tensor of predicted values
 */
/**
 * @doc {heading: 'Metrics', namespace: 'metrics'}
 */
export async function accuracy(
    labels: Tensor, predictions: Tensor): Promise<number> {
  assertShapesMatch(
      labels.shape, predictions.shape, 'Error computing accuracy.');

  const eq = labels.equal(predictions);
  const mean = eq.mean();

  const acc = (await mean.data())[0];

  dispose([eq, mean]);
  return acc;
}

/**
 * Computes per class accuracy between prediction and labels. Each value in
 * labels and predictions should correspond to some output class. It is assumed
 * that these values go from 0 to numClasses - 1.
 *
 * ```js
 * const labels = tf.tensor1d([0, 0, 1, 2, 2, 2]);
 * const predictions = tf.tensor1d([0, 0, 0, 2, 1, 1]);
 *
 * const result = await tfvis.metrics.perClassAccuracy(labels, predictions);
 * console.log(JSON.stringify(result, null, 2))
 * ```
 *
 * Returns an array of objects that each have an an `accuracy` and a `count`
 * property for each class.
 *
 *
 * @param labels 1D tensor of true values
 * @param predictions 1D tensor of predicted values
 * @param numClasses Number of distinct classes. Optional. If not passed in
 *  numClasses will equal the highest number in either labels or predictions
 *  plus 1
 */
/**
 * @doc {heading: 'Metrics', namespace: 'metrics'}
 */
export async function perClassAccuracy(
    labels: Tensor1D, predictions: Tensor1D,
    numClasses?: number): Promise<Array<{accuracy: number, count: number}>> {
  assert(labels.rank === 1, 'labels must be a 1D tensor');
  assert(predictions.rank === 1, 'predictions must be a 1D tensor');
  assert(
      labels.size === predictions.size,
      'labels and predictions must be the same length');

  if (numClasses == null) {
    numClasses = tidy(() => {
      return maximum(labels.max(), predictions.max()).dataSync()[0] + 1;
    });
  }

  return Promise.all([labels.data(), predictions.data()])
      .then(([labelsArray, predsArray]) => {
        // Per class total counts
        const counts: number[] = Array(numClasses).fill(0);
        // Per class accuracy
        const accuracy: number[] = Array(numClasses).fill(0);

        for (let i = 0; i < labelsArray.length; i++) {
          const label = labelsArray[i];
          const pred = predsArray[i];

          counts[label] += 1;
          if (label === pred) {
            accuracy[label] += 1;
          }
        }

        const results: Array<{accuracy: number, count: number}> = [];
        for (let i = 0; i < counts.length; i++) {
          results.push({
            count: counts[i],
            accuracy: counts[i] === 0 ? 0 : accuracy[i] / counts[i],
          });
        }

        return results;
      });
}
