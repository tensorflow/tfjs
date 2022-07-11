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

describeWithFlags('prod', ALL_ENVS, () => {
  it('basic', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const result = tf.prod(a);
    expectArraysClose(await result.data(), 0);
  });

  it('propagates NaNs', async () => {
    const a = tf.tensor2d([1, 2, 3, NaN, 0, 1], [3, 2]);
    expectArraysEqual(await tf.prod(a).data(), NaN);
  });

  it('prod over dtype int32', async () => {
    const a = tf.tensor1d([1, 5, 7, 3], 'int32');
    const prod = tf.prod(a);
    expectArraysEqual(await prod.data(), 105);
  });

  it('prod over dtype bool', async () => {
    const a = tf.tensor1d([true, false, false, true, true], 'bool');
    const prod = tf.prod(a);
    expectArraysEqual(await prod.data(), 0);
  });

  it('prods all values in 2D array with keep dim', async () => {
    const a = tf.tensor2d([1, 2, 3, 1, 0, 1], [3, 2]);
    const res = tf.prod(a, null, true /* keepDims */);

    expect(res.shape).toEqual([1, 1]);
    expectArraysClose(await res.data(), 0);
  });

  it('prods across axis=0 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 1, 0, 1], [3, 2]);
    const res = tf.prod(a, [0]);

    expect(res.shape).toEqual([2]);
    expectArraysClose(await res.data(), [0, 2]);
  });

  it('prods across axis=0 in 2D array, keepDims', async () => {
    const a = tf.tensor2d([1, 2, 3, 1, 0, 1], [3, 2]);
    const res = tf.prod(a, [0], true /* keepDims */);

    expect(res.shape).toEqual([1, 2]);
    expectArraysClose(await res.data(), [0, 2]);
  });

  it('prods across axis=1 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 1, 1, 1], [3, 2]);
    const res = tf.prod(a, [1]);

    expect(res.shape).toEqual([3]);
    expectArraysClose(await res.data(), [2, 3, 1]);
  });

  it('2D, axis=1 provided as number', async () => {
    const a = tf.tensor2d([1, 2, 3, 1, 1, 1], [2, 3]);
    const res = tf.prod(a, 1);

    expect(res.shape).toEqual([2]);
    expectArraysClose(await res.data(), [6, 1]);
  });

  it('2D, axis = -1 provided as number', async () => {
    const a = tf.tensor2d([1, 2, 3, 1, 1, 1], [2, 3]);
    const res = tf.prod(a, -1);

    expect(res.shape).toEqual([2]);
    expectArraysClose(await res.data(), [6, 1]);
  });

  it('prods across axis=0,1 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 1, 1, 1], [3, 2]);
    const res = tf.prod(a, [0, 1]);

    expect(res.shape).toEqual([]);
    expectArraysClose(await res.data(), [6]);
  });

  it('2D, axis=[-1,-2] in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 1, 1, 1], [3, 2]);
    const res = tf.prod(a, [-1, -2]);

    expect(res.shape).toEqual([]);
    expectArraysClose(await res.data(), [6]);
  });

  // Illustrative example from the C++ implementation for comparison. See:
  // https://github.com/tensorflow/tensorflow/blob/5dee95f8bafe3f342693d2135da451283b1fbaec/tensorflow/cc/gradients/math_grad.cc
  it('gradients: prod(multi-dimensional tensor with zeros)', async () => {
    const a = tf.tensor([
      [[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], [[3.0, 5.0], [0.0, 6.0], [5.0, 6.0]]
    ]);
    const dy = tf.tensor([[10, 10], [10, 10]]);

    const gradients = tf.grad(a => a.prod(1))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.array(), [
      [ // (axis 1 values) * (dy)
        [5 * 7 * 10, 6 * 8 * 10], [3 * 7 * 10, 4 * 8 * 10],
        [3 * 5 * 10, 4 * 6 * 10]
      ],
      [
        [0 * 5 * 10, 6 * 6 * 10], [3 * 5 * 10, 5 * 6 * 10],
        [3 * 0 * 10, 5 * 6 * 10]
      ]
    ]);
  });

  it('gradients: prod(reduce on 2 axes)', async () => {
    const a = tf.tensor([
      [[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], [[3.0, 5.0], [0.0, 6.0], [5.0, 6.0]]
    ]);
    const dy = tf.tensor([2, 2]);

    const gradients = tf.grad(a => a.prod([0, 1]))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');

    expectArraysClose(await gradients.array(), [
      [
        // (axis 0 values) * (axis 1 values) * (dy)
        [3 * 0 * 5 * (5 * 7) * 2, 5 * 6 * 6 * (6 * 8) * 2],
        [3 * 0 * 5 * (3 * 7) * 2, 5 * 6 * 6 * (4 * 8) * 2],
        [3 * 0 * 5 * (3 * 5) * 2, 5 * 6 * 6 * (4 * 6) * 2]
      ],
      [
        [3 * 5 * 7 * (0 * 5) * 2, 4 * 6 * 8 * (6 * 6) * 2],
        [3 * 5 * 7 * (3 * 5) * 2, 4 * 6 * 8 * (5 * 6) * 2],
        [3 * 5 * 7 * (3 * 0) * 2, 4 * 6 * 8 * (5 * 6) * 2]
      ]
    ]);
  });

  it('gradients: prod(tensor with zeros)', async () => {
    const a = tf.tensor2d([[1, 2, 0], [1, 2, 3]], [2, 3]);
    const dy = tf.tensor([12, 12]);

    const gradients = tf.grad(a => a.prod(1))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        await gradients.array(),
        [[0, 0, 1 * 2 * 12], [6 * 12, 3 * 12, 2 * 12]]);
  });

  it('gradients: prod(2d) with gradient broadcasting', async () => {
    const a = tf.tensor2d([[1, 2], [3, 4]], [2, 2]);
    const dy = tf.scalar(24);

    const gradients = tf.grad(a => a.prod())(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        await gradients.array(),
        [[2 * 3 * 4 * 24, 3 * 4 * 24], [4 * 2 * 24, 3 * 2 * 24]]);
  });

  it('gradients: prod(2d, axis=0) and varied gradient values', async () => {
    const a = tf.tensor2d([[1, 2], [3, 1], [1, 4]], [3, 2]);
    const dy = tf.tensor1d([3, 8]);
    const axis = 0;

    const gradients = tf.grad(a => a.prod(axis))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        await gradients.array(),
        [[3 * 3, 4 * 8], [1 * 3, 8 * 8], [3 * 3, 2 * 8]]);
  });

  it('gradients: prod(2d, axis=1)', async () => {
    const a = tf.tensor2d([[1, 2], [3, 1], [1, 4]], [3, 2]);
    const dy = tf.tensor1d([4, 3, 4]);
    const axis = 1;

    const gradients = tf.grad(a => a.prod(axis))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        await gradients.array(),
        [[2 * 4, 1 * 4], [1 * 3, 3 * 3], [4 * 4, 1 * 4]]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.prod({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'prod' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.prod([[1, 2], [3, 1], [1, 1]]);
    expectArraysClose(await result.data(), 6);
  });

  it('throws error for string tensor', () => {
    expect(() => tf.prod(['a']))
        .toThrowError(/Argument 'x' passed to 'prod' must be numeric tensor/);
  });
});
