/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import {expectArraysClose, expectArraysEqual} from '../test_util';

describeWithFlags('sum', ALL_ENVS, () => {
  it('basic', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const result = tf.sum(a);
    expectArraysClose(await result.data(), 7);
  });

  it('propagates NaNs', async () => {
    const a = tf.tensor2d([1, 2, 3, NaN, 0, 1], [3, 2]);
    expectArraysEqual(await tf.sum(a).data(), NaN);
  });

  it('sum over dtype int32', async () => {
    const a = tf.tensor1d([1, 5, 7, 3], 'int32');
    const sum = tf.sum(a);
    expectArraysEqual(await sum.data(), 16);
  });

  it('sum over dtype bool', async () => {
    const a = tf.tensor1d([true, false, false, true, true], 'bool');
    const sum = tf.sum(a);
    expectArraysEqual(await sum.data(), 3);
  });

  it('sums all values in 2D array with keep dim', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.sum(a, null, true /* keepDims */);

    expect(res.shape).toEqual([1, 1]);
    expectArraysClose(await res.data(), [7]);
  });

  it('sums across axis=0 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.sum(a, [0]);

    expect(res.shape).toEqual([2]);
    expectArraysClose(await res.data(), [4, 3]);
  });

  it('sums across axis=0 in 2D array, keepDims', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.sum(a, [0], true /* keepDims */);

    expect(res.shape).toEqual([1, 2]);
    expectArraysClose(await res.data(), [4, 3]);
  });

  it('sums across axis=1 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.sum(a, [1]);

    expect(res.shape).toEqual([3]);
    expectArraysClose(await res.data(), [3, 3, 1]);
  });

  it('2D, axis=1 provided as number', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [2, 3]);
    const res = tf.sum(a, 1);

    expect(res.shape).toEqual([2]);
    expectArraysClose(await res.data(), [6, 1]);
  });

  it('2D, axis = -1 provided as number', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [2, 3]);
    const res = tf.sum(a, -1);

    expect(res.shape).toEqual([2]);
    expectArraysClose(await res.data(), [6, 1]);
  });

  it('sums across axis=0,1 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.sum(a, [0, 1]);

    expect(res.shape).toEqual([]);
    expectArraysClose(await res.data(), [7]);
  });

  it('2D, axis=[-1,-2] in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.sum(a, [-1, -2]);

    expect(res.shape).toEqual([]);
    expectArraysClose(await res.data(), [7]);
  });

  it('4D, axis=[2, 2, 1], need permutation.', async () => {
    const a = tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]);
    const res = tf.sum(a, 0);

    expect(res.shape).toEqual([2, 2, 1]);
    expectArraysClose(await res.data(), [1, 2, 3, 4]);
  });

  it('gradients: sum(2d)', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const dy = tf.scalar(10);

    const gradients = tf.grad(a => a.sum())(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [10, 10, 10, 10, 10, 10]);
  });

  it('gradient with clones', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const dy = tf.scalar(10);

    const gradients = tf.grad(a => a.clone().sum().clone())(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [10, 10, 10, 10, 10, 10]);
  });

  it('gradients: sum(2d, axis=0)', async () => {
    const a = tf.tensor2d([[1, 2], [3, 0], [0, 1]], [3, 2]);
    const dy = tf.tensor1d([10, 20]);
    const axis = 0;

    const gradients = tf.grad(a => a.sum(axis))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [10, 20, 10, 20, 10, 20]);
  });

  it('gradients: sum(2d, axis=1)', async () => {
    const a = tf.tensor2d([[1, 2], [3, 0], [0, 1]], [3, 2]);
    const dy = tf.tensor1d([10, 20, 30]);
    const axis = 1;

    const gradients = tf.grad(a => a.sum(axis))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [10, 10, 20, 20, 30, 30]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.sum({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'sum' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.sum([[1, 2], [3, 0], [0, 1]]);
    expectArraysClose(await result.data(), 7);
  });

  it('throws error for string tensor', () => {
    expect(() => tf.sum(['a']))
        .toThrowError(/Argument 'x' passed to 'sum' must be numeric tensor/);
  });
});
