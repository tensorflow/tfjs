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

import * as tf from '../index';
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysClose} from '../test_util';

import {tensor1d, tensor2d, tensor3d} from './ops';

describeWithFlags('inTopKAsync', ALL_ENVS, async () => {
  it('predictions 2d array, targets 1d array, with default k', async () => {
    const predictions = tensor2d([[20, 10, 40, 30], [30, 50, -20, 10]]);
    const targets = tensor1d([2, 0]);
    const precision = await tf.inTopKAsync(predictions, targets);
    expect(precision.shape).toEqual([2]);
    expect(precision.dtype).toBe('bool');
    expectArraysClose(await precision.data(), [1, 0]);
  });

  it('predictions 2d array, targets 1d array, with k=2', async () => {
    const predictions = tensor2d([[20, 10, 40, 30], [30, 50, -20, 10]]);
    const targets = tensor1d([2, 0]);
    const k = 2;
    const precision = await tf.inTopKAsync(predictions, targets, k);
    expect(precision.shape).toEqual([2]);
    expect(precision.dtype).toBe('bool');
    expectArraysClose(await precision.data(), [1, 1]);
  });

  it('predictions 3d array, targets 2d array, with default k', async () => {
    const predictions =
        tensor3d([[[1, 5, 2], [4, 3, 6]], [[3, 2, 1], [1, 2, 3]]]);
    const targets = tensor2d([[1, 2], [0, 1]]);
    const precision = await tf.inTopKAsync(predictions, targets);
    expect(precision.shape).toEqual([2, 2]);
    expect(precision.dtype).toBe('bool');
    expectArraysClose(await precision.data(), [1, 1, 1, 0]);
  });

  it('predictions 3d array, targets 2d array, with k=2', async () => {
    const predictions =
        tensor3d([[[1, 5, 2], [4, 3, 6]], [[3, 2, 1], [1, 2, 3]]]);
    const targets = tensor2d([[1, 2], [0, 1]]);
    const k = 2;
    const precision = await tf.inTopKAsync(predictions, targets, k);
    expect(precision.shape).toEqual([2, 2]);
    expect(precision.dtype).toBe('bool');
    expectArraysClose(await precision.data(), [1, 1, 1, 1]);
  });

  it('lower-index element count first, with default k', async () => {
    const predictions = tensor2d([[1, 2, 2, 1]]);

    const targets1 = tensor1d([1]);
    const precision1 = await tf.inTopKAsync(predictions, targets1);
    expect(precision1.shape).toEqual([1]);
    expect(precision1.dtype).toBe('bool');
    expectArraysClose(await precision1.data(), [1]);

    const targets2 = tensor1d([2]);
    const precision2 = await tf.inTopKAsync(predictions, targets2);
    expect(precision2.shape).toEqual([1]);
    expect(precision2.dtype).toBe('bool');
    expectArraysClose(await precision2.data(), [0]);
  });

  it('accept tensor-like object, with default k', async () => {
    const predictions = [[20, 10, 40, 30], [30, 50, -20, 10]];
    const targets = [2, 0];
    const precision = await tf.inTopKAsync(predictions, targets);
    expect(precision.shape).toEqual([2]);
    expect(precision.dtype).toBe('bool');
    expectArraysClose(await precision.data(), [1, 0]);
  });

  it('doesnt leak tensors with tensor-like objects', async () => {
    const numTensors = tf.memory().numTensors;

    const predictions = [[20, 10, 40, 30], [30, 50, -20, 10]];
    const targets = [2, 0];
    const precision = await tf.inTopKAsync(predictions, targets);
    precision.dispose();

    expect(tf.memory().numTensors).toBe(numTensors);
  });

  it('throws when predictions_rank <2', async () => {
    const predictions = tensor1d([20, 10, 40, 30]);
    const targets = [2];

    // expect(...).toThrowError() does not support async functions.
    // See https://github.com/jasmine/jasmine/issues/1410
    try {
      await tf.inTopKAsync(predictions, targets);
      throw new Error('The line above should have thrown an error');
    } catch (ex) {
      expect(ex.message)
          .toEqual(
              'inTopK() expects the predictions to ' +
              'be of rank 2 or higher, but got 1');
    }
  });

  it('throws when prediction.rank != targets.rank + 1', async () => {
    const predictions = tensor2d([[20, 10, 40, 30], [30, 50, -20, 10]]);
    const targets = tensor2d([[0], [0]]);

    // expect(...).toThrowError() does not support async functions.
    // See https://github.com/jasmine/jasmine/issues/1410
    try {
      await tf.inTopKAsync(predictions, targets);
      throw new Error('The line above should have thrown an error');
    } catch (ex) {
      expect(ex.message)
          .toEqual(
              'predictions rank should be 1 larger than targets rank,' +
              ' but got predictions rank 2 and targets rank 2');
    }
  });

  it('throws when k > size of last dimension of predictions', async () => {
    const predictions = tensor2d([[20, 10, 40, 30], [30, 50, -20, 10]]);
    const targets = tensor1d([2, 0]);
    const k = 5;

    // expect(...).toThrowError() does not support async functions.
    // See https://github.com/jasmine/jasmine/issues/1410
    try {
      await tf.inTopKAsync(predictions, targets, k);
      throw new Error('The line above should have thrown an error');
    } catch (ex) {
      expect(ex.message)
          .toEqual(
              '\'k\' passed to inTopK() must be > 0 && <= the predictions ' +
              'last dimension (4), but got 5');
    }
  });
});
