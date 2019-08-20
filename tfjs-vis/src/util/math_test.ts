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

import * as tf from '@tensorflow/tfjs';

import {accuracy, arrayStats, confusionMatrix, perClassAccuracy, tensorStats} from './math';
import {DECIMAL_PLACES_TO_CHECK} from './utils';

//
// arrayStats
//
describe('arrayStats', () => {
  it('throws on non array input', () => {
    // @ts-ignore
    expect(() => arrayStats('string')).toThrow();
  });

  it('handles empty arrays', () => {
    const stats = arrayStats([]);
    expect(stats.max).toBe(undefined);
    expect(stats.min).toBe(undefined);
    expect(stats.numVals).toBe(0);
    expect(stats.numNans).toBe(0);
    expect(stats.numZeros).toBe(0);
  });

  it('computes correct stats', () => {
    const data = [2, 3, -400, 500, NaN, -800, 0, 0, 0];
    const stats = arrayStats(data);

    expect(stats.max).toBe(500);
    expect(stats.min).toBe(-800);
    expect(stats.numVals).toBe(9);
    expect(stats.numNans).toBe(1);
    expect(stats.numZeros).toBe(3);
  });

  it('computes correct stats — all negative', () => {
    const data = [-2, -3, -400, -500, NaN, -800];
    const stats = arrayStats(data);
    expect(stats.max).toBe(-2);
    expect(stats.min).toBe(-800);
    expect(stats.numVals).toBe(6);
    expect(stats.numNans).toBe(1);
    expect(stats.numZeros).toBe(0);
  });

  it('computes correct stats — all zeros', () => {
    const data = [0, 0, 0, 0];
    const stats = arrayStats(data);
    expect(stats.max).toBe(0);
    expect(stats.min).toBe(0);
    expect(stats.numVals).toBe(4);
    expect(stats.numNans).toBe(0);
    expect(stats.numZeros).toBe(4);
  });

  it('computes correct stats — all NaNs', () => {
    const data = [NaN, NaN, NaN, NaN];
    const stats = arrayStats(data);
    expect(isNaN(stats.max!)).toBe(true);
    expect(isNaN(stats.min!)).toBe(true);
    expect(stats.numVals).toBe(4);
    expect(stats.numNans).toBe(4);
    expect(stats.numZeros).toBe(0);
  });

  it('computes correct stats — some infs', () => {
    const data = [10, 4, Infinity, -Infinity, NaN];
    const stats = arrayStats(data);
    expect(stats.max).toBe(Infinity);
    expect(stats.min).toBe(-Infinity);
    expect(stats.numVals).toBe(5);
    expect(stats.numNans).toBe(1);
    expect(stats.numZeros).toBe(0);
    expect(stats.numInfs).toBe(2);
  });
});

//
// tensorStats
//
describe('tensorStats', () => {
  it('computes correct stats', async () => {
    const data = tf.tensor([2, 3, -400, 500, NaN, -800, 0, 0, 0]);
    const stats = await tensorStats(data);

    expect(stats.max).toBeCloseTo(500);
    expect(stats.min).toBeCloseTo(-800);
    expect(stats.numVals).toBe(9);
    expect(stats.numNans).toBe(1);
    expect(stats.numZeros).toBe(3);
  });

  it('computes correct stats — all negative', async () => {
    const data = tf.tensor([-2, -3, -400, -500, NaN, -800]);
    const stats = await tensorStats(data);
    expect(stats.max).toBeCloseTo(-2);
    expect(stats.min).toBeCloseTo(-800);
    expect(stats.numVals).toBe(6);
    expect(stats.numNans).toBe(1);
    expect(stats.numZeros).toBe(0);
  });

  it('computes correct stats — all zeros', async () => {
    const data = tf.tensor([0, 0, 0, 0]);
    const stats = await tensorStats(data);
    expect(stats.max).toBe(0);
    expect(stats.min).toBe(0);
    expect(stats.numVals).toBe(4);
    expect(stats.numNans).toBe(0);
    expect(stats.numZeros).toBe(4);
  });

  it('computes correct stats — all NaNs', async () => {
    const data = tf.tensor([NaN, NaN, NaN, NaN]);
    const stats = await tensorStats(data);
    expect(isNaN(stats.max!)).toBe(true);
    expect(isNaN(stats.min!)).toBe(true);
    expect(stats.numVals).toBe(4);
    expect(stats.numNans).toBe(4);
    expect(stats.numZeros).toBe(0);
  });

  it('computes correct stats — some infs', async () => {
    const data = tf.tensor([10, 4, Infinity, -Infinity, NaN]);
    const stats = await tensorStats(data);
    expect(stats.max).toBe(Infinity);
    expect(stats.min).toBe(-Infinity);
    expect(stats.numVals).toBe(5);
    expect(stats.numNans).toBe(1);
    expect(stats.numZeros).toBe(0);
    expect(stats.numInfs).toBe(2);
  });
});

//
// confusionMatrix
//
describe('confusionMatrix', () => {
  it('computes a confusion matrix', async () => {
    const labels = tf.tensor1d([1, 2, 4]);
    const predictions = tf.tensor1d([2, 2, 4]);

    const expected = [
      [0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0],
      [0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1],
    ];

    const result = await confusionMatrix(labels, predictions);
    expect(result).toEqual(expected);
  });

  it('computes a confusion matrix with explicit numClasses', async () => {
    const labels = tf.tensor1d([1, 2, 4]);
    const predictions = tf.tensor1d([2, 2, 4]);
    const numClasses = 6;

    const expected = [
      [0, 0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0],
      [0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 0],
    ];

    const result = await confusionMatrix(labels, predictions, numClasses);
    expect(result).toEqual(expected);
  });

  it('computes a confusion matrix with custom weights', async () => {
    const labels = tf.tensor1d([0, 1, 2, 3, 4]);
    const predictions = tf.tensor1d([0, 1, 2, 3, 4]);
    const weights = tf.tensor1d([0, 1, 2, 3, 4]);

    const expected = [
      [0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 0, 2, 0, 0],
      [0, 0, 0, 3, 0],
      [0, 0, 0, 0, 4],
    ];

    const result =
        await confusionMatrix(labels, predictions, undefined, weights);
    expect(result).toEqual(expected);
  });

  it('computes a confusion matrix where preds and labels do not intersect',
     async () => {
       const labels = tf.tensor1d([1, 1, 2, 3, 5, 1, 3, 6, 3, 1]);
       const predictions = tf.tensor1d([1, 1, 2, 3, 5, 6, 1, 2, 3, 4]);

       const expected = [
         [0, 0, 0, 0, 0, 0, 0],
         [0, 2, 0, 0, 1, 0, 1],
         [0, 0, 1, 0, 0, 0, 0],
         [0, 1, 0, 2, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0],
         [0, 0, 1, 0, 0, 0, 0],
       ];

       const result = await confusionMatrix(labels, predictions);
       expect(result).toEqual(expected);
     });

  it('computes a confusion matrix with multiple matches', async () => {
    const labels = tf.tensor1d([4, 5, 6]);
    const predictions = tf.tensor1d([1, 2, 3]);

    const expected = [
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0],
    ];

    const result = await confusionMatrix(labels, predictions);
    expect(result).toEqual(expected);
  });

  it('errors on non 1d label tensor', async () => {
    const labels = tf.tensor([1, 2, 4, 4], [2, 2]);
    const predictions = tf.tensor1d([2, 2, 4, 3]);

    let errorMessage;
    try {
      //@ts-ignore
      await confusionMatrix(labels, predictions);
    } catch (e) {
      errorMessage = e.message;
    }

    expect(errorMessage).toEqual('labels must be a 1D tensor');
  });

  it('errors on non 1d prediction tensor', async () => {
    const labels = tf.tensor1d([1, 2, 4, 4]);
    const predictions = tf.tensor([2, 2, 4, 3], [2, 2]);

    let errorMessage;
    try {
      //@ts-ignore
      await confusionMatrix(labels, predictions);
    } catch (e) {
      errorMessage = e.message;
    }

    expect(errorMessage).toEqual('predictions must be a 1D tensor');
  });

  it('errors on tensors of different lengths', async () => {
    const labels = tf.tensor1d([1, 2, 4]);
    const predictions = tf.tensor1d([2, 2, 4, 3, 6]);

    let errorMessage;
    try {
      //@ts-ignore
      await confusionMatrix(labels, predictions);
    } catch (e) {
      errorMessage = e.message;
    }

    expect(errorMessage)
        .toEqual('labels and predictions must be the same length');
  });
});

//
// accuracy
//
describe('accuracy', () => {
  it('computes accuracy', async () => {
    const labels = tf.tensor1d([1, 2, 4]);
    const predictions = tf.tensor1d([2, 2, 4]);

    const result = await accuracy(labels, predictions);
    expect(result).toBeCloseTo(2 / 3, DECIMAL_PLACES_TO_CHECK);
  });

  it('computes accuracy, no matches', async () => {
    const labels = tf.tensor1d([1, 2, 4]);
    const predictions = tf.tensor1d([5, 6, 8]);

    const result = await accuracy(labels, predictions);
    expect(result).toBeCloseTo(0, DECIMAL_PLACES_TO_CHECK);
  });

  it('computes accuracy, all matches', async () => {
    const labels = tf.tensor1d([1, 2, 3]);
    const predictions = tf.tensor1d([1, 2, 3]);

    const result = await accuracy(labels, predictions);
    expect(result).toBeCloseTo(1, DECIMAL_PLACES_TO_CHECK);
  });

  it('computes accuracy, tensor 2d', async () => {
    const labels = tf.tensor2d([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const predictions = tf.tensor2d([
      [1, 9],
      [3, 4],
      [5, 6],
    ]);

    const result = await accuracy(labels, predictions);
    expect(result).toBeCloseTo(5 / 6, DECIMAL_PLACES_TO_CHECK);
  });

  it('errors on tensors of different shapes', async () => {
    const labels = tf.tensor1d([1, 2, 4, 4]);
    const predictions = tf.tensor([2, 2, 4, 3], [2, 2]);

    let errorMessage;
    try {
      //@ts-ignore
      await accuracy(labels, predictions);
    } catch (e) {
      errorMessage = e.message;
    }

    expect(errorMessage)
        .toEqual('Error computing accuracy. Shapes 4 and 2,2 must match');
  });
});

//
// accuracy
//
describe('per class accuracy', () => {
  it('computes per class accuracy', async () => {
    const labels = tf.tensor1d([0, 0, 1, 2, 2, 2]);
    const predictions = tf.tensor1d([0, 0, 0, 2, 1, 1]);

    const result = await perClassAccuracy(labels, predictions);
    expect(result.map(d => d.accuracy)).toEqual([1, 0, 1 / 3]);
    expect(result.map(d => d.count)).toEqual([2, 1, 3]);
  });

  it('computes per class accuracy, no matches', async () => {
    const labels = tf.tensor1d([1, 1, 1, 1, 1]);
    const predictions = tf.tensor1d([0, 0, 0, 0, 0]);

    const result = await perClassAccuracy(labels, predictions);
    expect(result.map(d => d.accuracy)).toEqual([0, 0]);
    expect(result.map(d => d.count)).toEqual([0, 5]);
  });

  it('computes per class accuracy, all matches', async () => {
    const labels = tf.tensor1d([0, 1, 2, 3, 3, 3]);
    const predictions = tf.tensor1d([0, 1, 2, 3, 3, 3]);

    const result = await perClassAccuracy(labels, predictions);
    expect(result.map(d => d.accuracy)).toEqual([1, 1, 1, 1]);
    expect(result.map(d => d.count)).toEqual([1, 1, 1, 3]);
  });

  it('computes per class accuracy, explicit numClasses', async () => {
    const labels = tf.tensor1d([0, 1, 2, 2]);
    const predictions = tf.tensor1d([0, 1, 2, 1]);

    const result = await perClassAccuracy(labels, predictions, 5);
    expect(result.map(d => d.accuracy)).toEqual([1, 1, 0.5, 0, 0]);
    expect(result.map(d => d.count)).toEqual([1, 1, 2, 0, 0]);
  });

  it('errors on non 1d label tensor', async () => {
    const labels = tf.tensor([1, 2, 4, 4], [2, 2]);
    const predictions = tf.tensor1d([2, 2, 4, 3]);

    let errorMessage;
    try {
      //@ts-ignore
      await perClassAccuracy(labels, predictions);
    } catch (e) {
      errorMessage = e.message;
    }

    expect(errorMessage).toEqual('labels must be a 1D tensor');
  });

  it('errors on non 1d prediction tensor', async () => {
    const labels = tf.tensor1d([1, 2, 4, 4]);
    const predictions = tf.tensor([2, 2, 4, 3], [2, 2]);

    let errorMessage;
    try {
      //@ts-ignore
      await perClassAccuracy(labels, predictions);
    } catch (e) {
      errorMessage = e.message;
    }

    expect(errorMessage).toEqual('predictions must be a 1D tensor');
  });

  it('errors on tensors of different lengths', async () => {
    const labels = tf.tensor1d([1, 2, 4]);
    const predictions = tf.tensor1d([2, 2, 4, 3, 6]);

    let errorMessage;
    try {
      //@ts-ignore
      await perClassAccuracy(labels, predictions);
    } catch (e) {
      errorMessage = e.message;
    }

    expect(errorMessage)
        .toEqual('labels and predictions must be the same length');
  });
});
