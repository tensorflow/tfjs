/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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
import {describeWithFlags} from '../jasmine_util';
import {ALL_ENVS, expectArraysClose, expectArraysEqual, expectNumbersClose} from '../test_util';

import * as reduce_util from './reduce_util';

describeWithFlags('Reduction: min', ALL_ENVS, () => {
  it('Tensor1D', () => {
    const a = tf.tensor1d([3, -1, 0, 100, -7, 2]);
    expectNumbersClose(tf.min(a).get(), -7);
  });

  it('ignores NaNs', () => {
    const a = tf.tensor1d([3, NaN, 2]);
    expect(tf.min(a).get()).toEqual(2);
  });

  it('2D', () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    expectNumbersClose(tf.min(a).get(), -7);
  });

  it('2D axis=[0,1]', () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    expectNumbersClose(tf.min(a, [0, 1]).get(), -7);
  });

  it('2D, axis=0', () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    const r = tf.min(a, 0);

    expect(r.shape).toEqual([3]);
    expectArraysClose(r, [3, -7, 0]);
  });

  it('2D, axis=0, keepDims', () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    const r = tf.min(a, 0, true /* keepDims */);

    expect(r.shape).toEqual([1, 3]);
    expectArraysClose(r, [3, -7, 0]);
  });

  it('2D, axis=1 provided as a number', () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const r = tf.min(a, 1);
    expectArraysClose(r, [2, -7]);
  });

  it('2D, axis = -1 provided as a number', () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const r = tf.min(a, -1);
    expectArraysClose(r, [2, -7]);
  });

  it('2D, axis=[1]', () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const r = tf.min(a, [1]);
    expectArraysClose(r, [2, -7]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.min({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'min' must be a Tensor/);
  });

  it('accepts a tensor-like object', () => {
    expectNumbersClose(tf.min([3, -1, 0, 100, -7, 2]).get(), -7);
  });
});

describeWithFlags('Reduction: max', ALL_ENVS, () => {
  it('with one element dominating', () => {
    const a = tf.tensor1d([3, -1, 0, 100, -7, 2]);
    const r = tf.max(a);
    expectNumbersClose(r.get(), 100);
  });

  it('with all elements being the same', () => {
    const a = tf.tensor1d([3, 3, 3]);
    const r = tf.max(a);
    expectNumbersClose(r.get(), 3);
  });

  it('ignores NaNs', () => {
    expectNumbersClose(tf.max(tf.tensor1d([3, NaN, 2])).get(), 3);
  });

  it('2D', () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    expectNumbersClose(tf.max(a).get(), 100);
  });

  it('2D axis=[0,1]', () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    expectNumbersClose(tf.max(a, [0, 1]).get(), 100);
  });

  it('2D, axis=0', () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    const r = tf.max(a, [0]);

    expect(r.shape).toEqual([3]);
    expectArraysClose(r, [100, -1, 2]);
  });

  it('2D, axis=0, keepDims', () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    const r = tf.max(a, [0], true /* keepDims */);

    expect(r.shape).toEqual([1, 3]);
    expectArraysClose(r, [100, -1, 2]);
  });

  it('2D, axis=1 provided as a number', () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const r = tf.max(a, 1);
    expectArraysClose(r, [5, 100]);
  });

  it('2D, axis = -1 provided as a number', () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const r = tf.max(a, -1);
    expectArraysClose(r, [5, 100]);
  });

  it('2D, axis=[1]', () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const r = tf.max(a, [1]);
    expectArraysClose(r, [5, 100]);
  });

  it('6D, axis=[5]', () => {
    const a = tf.range(0, 64).reshape([2, 2, 2, 2, 2, 2]);
    const r = tf.max(a, [5]);
    const expectedResult = [
      1,  3,  5,  7,  9,  11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
      33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63
    ];
    expectArraysClose(r, expectedResult);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.max({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'max' must be a Tensor/);
  });

  it('accepts a tensor-like object', () => {
    const r = tf.max([3, -1, 0, 100, -7, 2]);
    expectNumbersClose(r.get(), 100);
  });
});

describeWithFlags('Reduction: argmax', ALL_ENVS, () => {
  it('Tensor1D', () => {
    const a = tf.tensor1d([1, 0, 3, 2]);
    const result = tf.argMax(a);
    expect(result.dtype).toBe('int32');
    expect(result.get()).toBe(2);
  });

  it('one value', () => {
    const a = tf.tensor1d([10]);
    const result = tf.argMax(a);
    expect(result.dtype).toBe('int32');
    expect(result.get()).toBe(0);
  });

  it('N > than parallelization threshold', () => {
    const n = reduce_util.PARALLELIZE_THRESHOLD * 2;
    const values = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      values[i] = i;
    }
    const a = tf.tensor1d(values);
    const result = tf.argMax(a);
    expect(result.dtype).toBe('int32');
    expect(result.get()).toBe(n - 1);
  });

  it('ignores NaNs', () => {
    const a = tf.tensor1d([0, 3, 5, NaN, 3]);
    const res = tf.argMax(a);
    expect(res.dtype).toBe('int32');
    expect(res.get()).toBe(2);
  });

  it('2D, no axis specified', () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    expectArraysEqual(tf.argMax(a), [1, 0, 1]);
  });

  it('2D, axis=0', () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    const r = tf.argMax(a, 0);

    expect(r.shape).toEqual([3]);
    expect(r.dtype).toBe('int32');
    expectArraysEqual(r, [1, 0, 1]);
  });

  it('2D, axis=1', () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const r = tf.argMax(a, 1);
    expect(r.dtype).toBe('int32');
    expectArraysEqual(r, [2, 0]);
  });

  it('2D, axis = -1', () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const r = tf.argMax(a, -1);
    expect(r.dtype).toBe('int32');
    expectArraysEqual(r, [2, 0]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.argMax({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'argMax' must be a Tensor/);
  });

  it('accepts a tensor-like object', () => {
    const result = tf.argMax([1, 0, 3, 2]);
    expect(result.dtype).toBe('int32');
    expect(result.get()).toBe(2);
  });
});

describeWithFlags('Reduction: argmin', ALL_ENVS, () => {
  it('Tensor1D', () => {
    const a = tf.tensor1d([1, 0, 3, 2]);
    const result = tf.argMin(a);
    expect(result.get()).toBe(1);
  });

  it('one value', () => {
    const a = tf.tensor1d([10]);
    const result = tf.argMin(a);
    expect(result.get()).toBe(0);
  });

  it('N > than parallelization threshold', () => {
    const n = reduce_util.PARALLELIZE_THRESHOLD * 2;
    const values = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      values[i] = n - i;
    }
    const a = tf.tensor1d(values);
    const result = tf.argMin(a);
    expect(result.dtype).toBe('int32');
    expect(result.get()).toBe(n - 1);
  });

  it('ignores NaNs', () => {
    const a = tf.tensor1d([5, 0, NaN, -1, 3]);
    const res = tf.argMin(a);
    expect(res.get()).toBe(3);
  });

  it('2D, no axis specified', () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    expectArraysEqual(tf.argMin(a), [0, 1, 0]);
  });

  it('2D, axis=0', () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    const r = tf.argMin(a, 0);

    expect(r.shape).toEqual([3]);
    expect(r.dtype).toBe('int32');
    expectArraysEqual(r, [0, 1, 0]);
  });

  it('2D, axis=1', () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, -8], [2, 3]);
    const r = tf.argMin(a, 1);
    expectArraysEqual(r, [1, 2]);
  });

  it('2D, axis = -1', () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, -8], [2, 3]);
    const r = tf.argMin(a, -1);
    expectArraysEqual(r, [1, 2]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.argMin({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'argMin' must be a Tensor/);
  });

  it('accepts a tensor-like object', () => {
    const result = tf.argMin([1, 0, 3, 2]);
    expect(result.get()).toBe(1);
  });
});

describeWithFlags('Reduction: logSumExp', ALL_ENVS, () => {
  it('0', () => {
    const a = tf.scalar(0);
    const result = tf.logSumExp(a);
    expectNumbersClose(result.get(), 0);
  });

  it('basic', () => {
    const a = tf.tensor1d([1, 2, -3]);
    const result = tf.logSumExp(a);

    expectNumbersClose(
        result.get(), Math.log(Math.exp(1) + Math.exp(2) + Math.exp(-3)));
  });

  it('propagates NaNs', () => {
    const a = tf.tensor1d([1, 2, NaN]);
    const result = tf.logSumExp(a);
    expect(result.get()).toEqual(NaN);
  });

  it('axes=0 in 2D array', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const r = tf.logSumExp(a, [0]);

    expect(r.shape).toEqual([2]);
    const expected = [
      Math.log(Math.exp(1) + Math.exp(3) + Math.exp(0)),
      Math.log(Math.exp(2) + Math.exp(0) + Math.exp(1))
    ];
    expectArraysClose(r, expected);
  });

  it('axes=0 in 2D array, keepDims', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const r = tf.logSumExp(a, [0], true /* keepDims */);

    expect(r.shape).toEqual([1, 2]);
    const expected = [
      Math.log(Math.exp(1) + Math.exp(3) + Math.exp(0)),
      Math.log(Math.exp(2) + Math.exp(0) + Math.exp(1))
    ];
    expectArraysClose(r, expected);
  });

  it('axes=1 in 2D array', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.logSumExp(a, [1]);

    expect(res.shape).toEqual([3]);
    const expected = [
      Math.log(Math.exp(1) + Math.exp(2)),
      Math.log(Math.exp(3) + Math.exp(0)),
      Math.log(Math.exp(0) + Math.exp(1)),
    ];
    expectArraysClose(res, expected);
  });

  it('axes = -1 in 2D array', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.logSumExp(a, -1);

    expect(res.shape).toEqual([3]);
    const expected = [
      Math.log(Math.exp(1) + Math.exp(2)),
      Math.log(Math.exp(3) + Math.exp(0)),
      Math.log(Math.exp(0) + Math.exp(1)),
    ];
    expectArraysClose(res, expected);
  });

  it('2D, axes=1 provided as a single digit', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [2, 3]);
    const res = tf.logSumExp(a, 1);

    expect(res.shape).toEqual([2]);
    const expected = [
      Math.log(Math.exp(1) + Math.exp(2) + Math.exp(3)),
      Math.log(Math.exp(0) + Math.exp(0) + Math.exp(1))
    ];
    expectArraysClose(res, expected);
  });

  it('axes=0,1 in 2D array', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.logSumExp(a, [0, 1]);

    expect(res.shape).toEqual([]);
    const expected = [Math.log(
        Math.exp(1) + Math.exp(2) + Math.exp(3) + Math.exp(0) + Math.exp(0) +
        Math.exp(1))];
    expectArraysClose(res, expected);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.logSumExp({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'logSumExp' must be a Tensor/);
  });

  it('accepts a tensor-like object', () => {
    const result = tf.logSumExp([1, 2, -3]);
    expectNumbersClose(
        result.get(), Math.log(Math.exp(1) + Math.exp(2) + Math.exp(-3)));
  });
});

describeWithFlags('Reduction: sum', ALL_ENVS, () => {
  it('basic', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const result = tf.sum(a);
    expectNumbersClose(result.get(), 7);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor2d([1, 2, 3, NaN, 0, 1], [3, 2]);
    expect(tf.sum(a).get()).toEqual(NaN);
  });

  it('sum over dtype int32', () => {
    const a = tf.tensor1d([1, 5, 7, 3], 'int32');
    const sum = tf.sum(a);
    expect(sum.get()).toBe(16);
  });

  it('sum over dtype bool', () => {
    const a = tf.tensor1d([true, false, false, true, true], 'bool');
    const sum = tf.sum(a);
    expect(sum.get()).toBe(3);
  });

  it('sums all values in 2D array with keep dim', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.sum(a, null, true /* keepDims */);

    expect(res.shape).toEqual([1, 1]);
    expectArraysClose(res, [7]);
  });

  it('sums across axis=0 in 2D array', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.sum(a, [0]);

    expect(res.shape).toEqual([2]);
    expectArraysClose(res, [4, 3]);
  });

  it('sums across axis=0 in 2D array, keepDims', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.sum(a, [0], true /* keepDims */);

    expect(res.shape).toEqual([1, 2]);
    expectArraysClose(res, [4, 3]);
  });

  it('sums across axis=1 in 2D array', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.sum(a, [1]);

    expect(res.shape).toEqual([3]);
    expectArraysClose(res, [3, 3, 1]);
  });

  it('2D, axis=1 provided as number', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [2, 3]);
    const res = tf.sum(a, 1);

    expect(res.shape).toEqual([2]);
    expectArraysClose(res, [6, 1]);
  });

  it('2D, axis = -1 provided as number', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [2, 3]);
    const res = tf.sum(a, -1);

    expect(res.shape).toEqual([2]);
    expectArraysClose(res, [6, 1]);
  });

  it('sums across axis=0,1 in 2D array', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.sum(a, [0, 1]);

    expect(res.shape).toEqual([]);
    expectArraysClose(res, [7]);
  });

  it('2D, axis=[-1,-2] in 2D array', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.sum(a, [-1, -2]);

    expect(res.shape).toEqual([]);
    expectArraysClose(res, [7]);
  });

  it('gradients: sum(2d)', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const dy = tf.scalar(10);

    const gradients = tf.grad(a => a.sum())(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [10, 10, 10, 10, 10, 10]);
  });

  it('gradients: sum(2d, axis=0)', () => {
    const a = tf.tensor2d([[1, 2], [3, 0], [0, 1]], [3, 2]);
    const dy = tf.tensor1d([10, 20]);
    const axis = 0;

    const gradients = tf.grad(a => a.sum(axis))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [10, 20, 10, 20, 10, 20]);
  });

  it('gradients: sum(2d, axis=1)', () => {
    const a = tf.tensor2d([[1, 2], [3, 0], [0, 1]], [3, 2]);
    const dy = tf.tensor1d([10, 20, 30]);
    const axis = 1;

    const gradients = tf.grad(a => a.sum(axis))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(gradients, [10, 10, 20, 20, 30, 30]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.sum({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'sum' must be a Tensor/);
  });

  it('accepts a tensor-like object', () => {
    const result = tf.sum([[1, 2], [3, 0], [0, 1]]);
    expectNumbersClose(result.get(), 7);
  });
});

describeWithFlags('Reduction: mean', ALL_ENVS, () => {
  it('basic', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const r = tf.mean(a);

    expect(r.dtype).toBe('float32');
    expectNumbersClose(r.get(), 7 / 6);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor2d([1, 2, 3, NaN, 0, 1], [3, 2]);
    const r = tf.mean(a);

    expect(r.dtype).toBe('float32');
    expect(r.get()).toEqual(NaN);
  });

  it('mean(int32) => float32', () => {
    const a = tf.tensor1d([1, 5, 7, 3], 'int32');
    const r = tf.mean(a);

    expect(r.dtype).toBe('float32');
    expectNumbersClose(r.get(), 4);
  });

  it('mean(bool) => float32', () => {
    const a = tf.tensor1d([true, false, false, true, true], 'bool');
    const r = tf.mean(a);

    expect(r.dtype).toBe('float32');
    expectNumbersClose(r.get(), 3 / 5);
  });

  it('2D array with keep dim', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.mean(a, null, true /* keepDims */);

    expect(res.shape).toEqual([1, 1]);
    expect(res.dtype).toBe('float32');
    expectArraysClose(res, [7 / 6]);
  });

  it('axis=0 in 2D array', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.mean(a, [0]);

    expect(res.shape).toEqual([2]);
    expect(res.dtype).toBe('float32');
    expectArraysClose(res, [4 / 3, 1]);
  });

  it('axis=0 in 2D array, keepDims', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.mean(a, [0], true /* keepDims */);

    expect(res.shape).toEqual([1, 2]);
    expect(res.dtype).toBe('float32');
    expectArraysClose(res, [4 / 3, 1]);
  });

  it('axis=1 in 2D array', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.mean(a, [1]);

    expect(res.dtype).toBe('float32');
    expect(res.shape).toEqual([3]);
    expectArraysClose(res, [1.5, 1.5, 0.5]);
  });

  it('axis = -1 in 2D array', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.mean(a, [-1]);

    expect(res.dtype).toBe('float32');
    expect(res.shape).toEqual([3]);
    expectArraysClose(res, [1.5, 1.5, 0.5]);
  });

  it('2D, axis=1 provided as number', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [2, 3]);
    const res = tf.mean(a, 1);

    expect(res.shape).toEqual([2]);
    expect(res.dtype).toBe('float32');
    expectArraysClose(res, [2, 1 / 3]);
  });

  it('axis=0,1 in 2D array', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.mean(a, [0, 1]);

    expect(res.shape).toEqual([]);
    expect(res.dtype).toBe('float32');
    expectArraysClose(res, [7 / 6]);
  });

  it('gradients', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const dy = tf.scalar(1.5);

    const da = tf.grad(a => a.mean())(a, dy);

    expect(da.shape).toEqual(a.shape);
    expectArraysClose(da, [
      dy.get() / a.size, dy.get() / a.size, dy.get() / a.size,
      dy.get() / a.size, dy.get() / a.size, dy.get() / a.size
    ]);
  });

  it('gradients throws for defined axis', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const dy = tf.scalar(1.5);

    expect(() => tf.grad(a => a.mean(1))(a, dy)).toThrowError();
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.mean({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'mean' must be a Tensor/);
  });

  it('accepts a tensor-like object', () => {
    const r = tf.mean([[1, 2, 3], [0, 0, 1]]);

    expect(r.dtype).toBe('float32');
    expectNumbersClose(r.get(), 7 / 6);
  });
});

describeWithFlags('Reduction: moments', ALL_ENVS, () => {
  it('basic', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const {mean, variance} = tf.moments(a);

    expect(mean.dtype).toBe('float32');
    expect(variance.dtype).toBe('float32');
    expectNumbersClose(mean.get(), 7 / 6);
    expectNumbersClose(variance.get(), 1.1389);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor2d([1, 2, 3, NaN, 0, 1], [3, 2]);
    const {mean, variance} = tf.moments(a);

    expect(mean.dtype).toBe('float32');
    expect(variance.dtype).toBe('float32');
    expect(mean.get()).toEqual(NaN);
    expect(variance.get()).toEqual(NaN);
  });

  it('moments(int32) => float32', () => {
    const a = tf.tensor1d([1, 5, 7, 3], 'int32');
    const {mean, variance} = tf.moments(a);

    expect(mean.dtype).toBe('float32');
    expect(variance.dtype).toBe('float32');
    expectNumbersClose(mean.get(), 4);
    expectNumbersClose(variance.get(), 5);
  });

  it('moments(bool) => float32', () => {
    const a = tf.tensor1d([true, false, false, true, true], 'bool');
    const {mean, variance} = tf.moments(a);

    expect(mean.dtype).toBe('float32');
    expect(variance.dtype).toBe('float32');
    expectNumbersClose(mean.get(), 3 / 5);
    expectNumbersClose(variance.get(), 0.23999998);
  });

  it('2D array with keep dim', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const {mean, variance} = tf.moments(a, null, true /* keepDims */);

    expect(mean.shape).toEqual([1, 1]);
    expect(mean.dtype).toBe('float32');
    expect(variance.shape).toEqual([1, 1]);
    expect(variance.dtype).toBe('float32');
    expectArraysClose(mean, [7 / 6]);
    expectArraysClose(variance, [1.138889]);
  });

  it('axis=0 in 2D array', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const {mean, variance} = tf.moments(a, [0]);

    expect(mean.shape).toEqual([2]);
    expect(mean.dtype).toBe('float32');
    expect(variance.shape).toEqual([2]);
    expect(variance.dtype).toBe('float32');
    expectArraysClose(mean, [4 / 3, 1]);
    expectArraysClose(variance, [1.556, 2 / 3]);
  });

  it('axis=1 in 2D array', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const {mean, variance} = tf.moments(a, [1]);

    expect(mean.dtype).toBe('float32');
    expect(mean.shape).toEqual([3]);
    expect(variance.dtype).toBe('float32');
    expect(variance.shape).toEqual([3]);
    expectArraysClose(mean, [1.5, 1.5, 0.5]);
    expectArraysClose(variance, [0.25, 2.25, 0.25]);
  });

  it('2D, axis=1 provided as number', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [2, 3]);
    const {mean, variance} = tf.moments(a, 1);

    expect(mean.shape).toEqual([2]);
    expect(mean.dtype).toBe('float32');
    expect(variance.shape).toEqual([2]);
    expect(variance.dtype).toBe('float32');
    expectArraysClose(mean, [2, 1 / 3]);
    expectArraysClose(variance, [2 / 3, 0.222]);
  });

  it('2D, axis=-1 provided as number', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [2, 3]);
    const {mean, variance} = tf.moments(a, -1);

    expect(mean.shape).toEqual([2]);
    expect(mean.dtype).toBe('float32');
    expect(variance.shape).toEqual([2]);
    expect(variance.dtype).toBe('float32');
    expectArraysClose(mean, [2, 1 / 3]);
    expectArraysClose(variance, [2 / 3, 0.222]);
  });

  it('axis=0,1 in 2D array', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const {mean, variance} = tf.moments(a, [0, 1]);

    expect(mean.shape).toEqual([]);
    expect(mean.dtype).toBe('float32');
    expect(variance.shape).toEqual([]);
    expect(variance.dtype).toBe('float32');
    expectArraysClose(mean, [7 / 6]);
    expectArraysClose(variance, [1.1389]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.moments({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'moments' must be a Tensor/);
  });

  it('accepts a tensor-like object', () => {
    const {mean, variance} = tf.moments([1, 2, 3, 0, 0, 1]);

    expect(mean.dtype).toBe('float32');
    expect(variance.dtype).toBe('float32');
    expectNumbersClose(mean.get(), 7 / 6);
    expectNumbersClose(variance.get(), 1.1389);
  });
});

describeWithFlags('Reduction: norm', ALL_ENVS, () => {
  it('scalar norm', () => {
    const a = tf.scalar(-22.0);
    const norm = tf.norm(a);

    expect(norm.dtype).toBe('float32');
    expectNumbersClose(norm.get(), 22);
  });

  it('vector inf norm', () => {
    const a = tf.tensor1d([1, -2, 3, -4]);
    const norm = tf.norm(a, Infinity);

    expect(norm.dtype).toBe('float32');
    expectNumbersClose(norm.get(), 4);
  });

  it('vector -inf norm', () => {
    const a = tf.tensor1d([1, -2, 3, -4]);
    const norm = tf.norm(a, -Infinity);

    expect(norm.dtype).toBe('float32');
    expectNumbersClose(norm.get(), 1);
  });

  it('vector 1 norm', () => {
    const a = tf.tensor1d([1, -2, 3, -4]);
    const norm = tf.norm(a, 1);

    expect(norm.dtype).toBe('float32');
    expectNumbersClose(norm.get(), 10);
  });

  it('vector euclidean norm', () => {
    const a = tf.tensor1d([1, -2, 3, -4]);
    const norm = tf.norm(a, 'euclidean');

    expect(norm.dtype).toBe('float32');
    expectNumbersClose(norm.get(), 5.4772);
  });

  it('vector 2-norm', () => {
    const a = tf.tensor1d([1, -2, 3, -4]);
    const norm = tf.norm(a, 2);

    expect(norm.dtype).toBe('float32');
    expectNumbersClose(norm.get(), 5.4772);
  });

  it('vector >2-norm to throw error', () => {
    const a = tf.tensor1d([1, -2, 3, -4]);
    expect(() => tf.norm(a, 3)).toThrowError();
  });

  it('matrix inf norm', () => {
    const a = tf.tensor2d([1, 2, -3, 1, 0, 1], [3, 2]);
    const norm = tf.norm(a, Infinity, [0, 1]);

    expect(norm.dtype).toBe('float32');
    expectNumbersClose(norm.get(), 4);
  });

  it('matrix -inf norm', () => {
    const a = tf.tensor2d([1, 2, -3, 1, 0, 1], [3, 2]);
    const norm = tf.norm(a, -Infinity, [0, 1]);

    expect(norm.dtype).toBe('float32');
    expectNumbersClose(norm.get(), 1);
  });

  it('matrix 1 norm', () => {
    const a = tf.tensor2d([1, 2, -3, 1, 1, 1], [3, 2]);
    const norm = tf.norm(a, 1, [0, 1]);

    expect(norm.dtype).toBe('float32');
    expectNumbersClose(norm.get(), 5);
  });

  it('matrix euclidean norm', () => {
    const a = tf.tensor2d([1, 2, -3, 1, 1, 1], [3, 2]);
    const norm = tf.norm(a, 'euclidean', [0, 1]);

    expect(norm.dtype).toBe('float32');
    expectNumbersClose(norm.get(), 4.123);
  });

  it('matrix fro norm', () => {
    const a = tf.tensor2d([1, 2, -3, 1, 1, 1], [3, 2]);
    const norm = tf.norm(a, 'fro', [0, 1]);

    expect(norm.dtype).toBe('float32');
    expectNumbersClose(norm.get(), 4.123);
  });

  it('matrix other norm to throw error', () => {
    const a = tf.tensor2d([1, 2, -3, 1, 1, 1], [3, 2]);
    expect(() => tf.norm(a, 2, [0, 1])).toThrowError();
  });

  it('propagates NaNs for norm', () => {
    const a = tf.tensor2d([1, 2, 3, NaN, 0, 1], [3, 2]);
    const norm = tf.norm(a);

    expect(norm.dtype).toBe('float32');
    expect(norm.get()).toEqual(NaN);
  });

  it('axis=null in 2D array norm', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const norm = tf.norm(a, Infinity);

    expect(norm.shape).toEqual([]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(norm, [3]);
  });

  it('2D array norm with keep dim', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const norm = tf.norm(a, Infinity, null, true /* keepDims */);

    expect(norm.shape).toEqual([1, 1]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(norm, [3]);
  });

  it('axis=0 in 2D array norm', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const norm = tf.norm(a, Infinity, [0]);

    expect(norm.shape).toEqual([2]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(norm, [3, 2]);
  });

  it('axis=1 in 2D array norm', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const norm = tf.norm(a, Infinity, [1]);

    expect(norm.dtype).toBe('float32');
    expect(norm.shape).toEqual([3]);
    expectArraysClose(norm, [2, 3, 1]);
  });

  it('axis=1 keepDims in 2D array norm', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const norm = tf.norm(a, Infinity, [1], true);

    expect(norm.dtype).toBe('float32');
    expect(norm.shape).toEqual([3, 1]);
    expectArraysClose(norm, [2, 3, 1]);
  });

  it('2D norm with axis=1 provided as number', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [2, 3]);
    const norm = tf.norm(a, Infinity, 1);

    expect(norm.shape).toEqual([2]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(norm, [3, 1]);
  });

  it('axis=0,1 in 2D array norm', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const norm = tf.norm(a, Infinity, [0, 1]);

    expect(norm.shape).toEqual([]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(norm, [3]);
  });

  it('axis=0,1 keepDims in 2D array norm', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const norm = tf.norm(a, Infinity, [0, 1], true);

    expect(norm.shape).toEqual([1, 1]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(norm, [3]);
  });

  it('3D norm axis=0,1, matrix inf norm', () => {
    const a = tf.tensor3d([1, 2, -3, 1, 0, 1], [3, 2, 1]);
    const norm = tf.norm(a, Infinity, [0, 1]);

    expect(norm.shape).toEqual([1]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(norm, [4]);
  });

  it('axis=0,1 keepDims in 3D array norm', () => {
    const a = tf.tensor3d([1, 2, 3, 0, 0, 1], [3, 2, 1]);
    const norm = tf.norm(a, Infinity, [0, 1], true);

    expect(norm.shape).toEqual([1, 1, 1]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(norm, [3]);
  });

  it('axis=0,1 keepDims in 3D array norm', () => {
    const a = tf.tensor3d([1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 1], [3, 2, 2]);
    const norm = tf.norm(a, Infinity, [0, 1], true);

    expect(norm.shape).toEqual([1, 1, 2]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(norm, [4, 3]);
  });

  it('axis=null in 3D array norm', () => {
    const a = tf.tensor3d([1, 2, 3, 0, 0, 1], [3, 2, 1]);
    const norm = tf.norm(a, Infinity);

    expect(norm.shape).toEqual([]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(norm, [3]);
  });

  it('axis=null in 4D array norm', () => {
    const a = tf.tensor4d([1, 2, 3, 0, 0, 1], [3, 2, 1, 1]);
    const norm = tf.norm(a, Infinity);

    expect(norm.shape).toEqual([]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(norm, [3]);
  });

  it('axis=0,1 in 4D array norm', () => {
    const a = tf.tensor4d(
        [
          1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 1,
          1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 1
        ],
        [3, 2, 2, 2]);
    const norm = tf.norm(a, Infinity, [0, 1]);

    expect(norm.shape).toEqual([2, 2]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(norm, [4, 3, 4, 3]);
  });

  it('axis=0,1 in 4D array norm', () => {
    const a = tf.tensor4d(
        [
          1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 1,
          1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 1
        ],
        [3, 2, 2, 2]);
    const norm = tf.norm(a, Infinity, [0, 1], true);

    expect(norm.shape).toEqual([1, 1, 2, 2]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(norm, [4, 3, 4, 3]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.norm({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'norm' must be a Tensor/);
  });

  it('accepts a tensor-like object', () => {
    const norm = tf.norm([1, -2, 3, -4], 1);

    expect(norm.dtype).toBe('float32');
    expectNumbersClose(norm.get(), 10);
  });
});

describeWithFlags('Reduction: all', ALL_ENVS, () => {
  it('Tensor1D', () => {
    let a = tf.tensor1d([0, 0, 0], 'bool');
    expectNumbersClose(tf.all(a).get(), 0);

    a = tf.tensor1d([1, 0, 1], 'bool');
    expectNumbersClose(tf.all(a).get(), 0);

    a = tf.tensor1d([1, 1, 1], 'bool');
    expectNumbersClose(tf.all(a).get(), 1);
  });

  it('ignores NaNs', () => {
    const a = tf.tensor1d([1, NaN, 1], 'bool');
    expect(tf.all(a).get()).toEqual(1);
  });

  it('2D', () => {
    const a = tf.tensor2d([1, 1, 0, 0], [2, 2], 'bool');
    expectNumbersClose(tf.all(a).get(), 0);
  });

  it('2D axis=[0,1]', () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    expectNumbersClose(tf.all(a, [0, 1]).get(), 0);
  });

  it('2D, axis=0', () => {
    const a = tf.tensor2d([1, 1, 0, 0], [2, 2], 'bool');
    let r = tf.all(a, 0);

    expect(r.shape).toEqual([2]);
    expectArraysClose(r, [0, 0]);

    r = tf.all(a, 1);

    expect(r.shape).toEqual([2]);
    expectArraysClose(r, [1, 0]);
  });

  it('2D, axis=0, keepDims', () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    const r = a.all(0, true /* keepDims */);

    expect(r.shape).toEqual([1, 3]);
    expectArraysClose(r, [0, 1, 0]);
  });

  it('2D, axis=1 provided as a number', () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    const r = tf.all(a, 1);
    expectArraysClose(r, [0, 0]);
  });

  it('2D, axis = -1 provided as a number', () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    const r = tf.all(a, -1);
    expectArraysClose(r, [0, 0]);
  });

  it('2D, axis=[1]', () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    const r = tf.all(a, [1]);
    expectArraysClose(r, [0, 0]);
  });

  it('throws when dtype is not boolean', () => {
    const a = tf.tensor2d([1, 1, 0, 0], [2, 2]);
    expect(() => tf.all(a)).toThrowError(/Error Tensor must be of type bool/);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.all({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'all' must be a Tensor/);
  });

  it('accepts a tensor-like object', () => {
    const a = [0, 0, 0];
    expectNumbersClose(tf.all(a).get(), 0);
  });
});

describeWithFlags('Reduction: any', ALL_ENVS, () => {
  it('Tensor1D', () => {
    let a = tf.tensor1d([0, 0, 0], 'bool');
    expectNumbersClose(tf.any(a).get(), 0);

    a = tf.tensor1d([1, 0, 1], 'bool');
    expectNumbersClose(tf.any(a).get(), 1);

    a = tf.tensor1d([1, 1, 1], 'bool');
    expectNumbersClose(tf.any(a).get(), 1);
  });

  it('ignores NaNs', () => {
    const a = tf.tensor1d([1, NaN, 0], 'bool');
    expect(tf.any(a).get()).toEqual(1);
  });

  it('2D', () => {
    const a = tf.tensor2d([1, 1, 0, 0], [2, 2], 'bool');
    expectNumbersClose(tf.any(a).get(), 1);
  });

  it('2D axis=[0,1]', () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    expectNumbersClose(tf.any(a, [0, 1]).get(), 1);
  });

  it('2D, axis=0', () => {
    const a = tf.tensor2d([1, 1, 0, 0], [2, 2], 'bool');
    let r = tf.any(a, 0);

    expect(r.shape).toEqual([2]);
    expectArraysClose(r, [1, 1]);

    r = tf.any(a, 1);

    expect(r.shape).toEqual([2]);
    expectArraysClose(r, [1, 0]);
  });

  it('2D, axis=0, keepDims', () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    const r = a.any(0, true /* keepDims */);

    expect(r.shape).toEqual([1, 3]);
    expectArraysClose(r, [1, 1, 0]);
  });

  it('2D, axis=1 provided as a number', () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    const r = tf.any(a, 1);
    expectArraysClose(r, [1, 1]);
  });

  it('2D, axis = -1 provided as a number', () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    const r = tf.any(a, -1);
    expectArraysClose(r, [1, 1]);
  });

  it('2D, axis=[1]', () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    const r = tf.any(a, [1]);
    expectArraysClose(r, [1, 1]);
  });

  it('throws when dtype is not boolean', () => {
    const a = tf.tensor2d([1, 1, 0, 0], [2, 2]);
    expect(() => tf.any(a)).toThrowError(/Error Tensor must be of type bool/);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.any({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'any' must be a Tensor/);
  });

  it('accepts a tensor-like object', () => {
    const a = [0, 0, 0];
    expectNumbersClose(tf.any(a).get(), 0);
  });
});
