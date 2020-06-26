/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

describeWithFlags('mean', ALL_ENVS, () => {
  it('basic', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const r = tf.mean(a);

    expect(r.dtype).toBe('float32');
    expectArraysClose(await r.data(), 7 / 6);
  });

  it('propagates NaNs', async () => {
    const a = tf.tensor2d([1, 2, 3, NaN, 0, 1], [3, 2]);
    const r = tf.mean(a);

    expect(r.dtype).toBe('float32');
    expectArraysEqual(await r.data(), NaN);
  });

  it('mean(int32) => float32', async () => {
    const a = tf.tensor1d([1, 5, 7, 3], 'int32');
    const r = tf.mean(a);

    expect(r.dtype).toBe('float32');
    expectArraysClose(await r.data(), 4);
  });

  it('mean(bool) => float32', async () => {
    const a = tf.tensor1d([true, false, false, true, true], 'bool');
    const r = tf.mean(a);

    expect(r.dtype).toBe('float32');
    expectArraysClose(await r.data(), 3 / 5);
  });

  it('2D array with keep dim', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.mean(a, null, true /* keepDims */);

    expect(res.shape).toEqual([1, 1]);
    expect(res.dtype).toBe('float32');
    expectArraysClose(await res.data(), [7 / 6]);
  });

  it('axis=0 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.mean(a, [0]);

    expect(res.shape).toEqual([2]);
    expect(res.dtype).toBe('float32');
    expectArraysClose(await res.data(), [4 / 3, 1]);
  });

  it('axis=0 in 2D array, keepDims', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.mean(a, [0], true /* keepDims */);

    expect(res.shape).toEqual([1, 2]);
    expect(res.dtype).toBe('float32');
    expectArraysClose(await res.data(), [4 / 3, 1]);
  });

  it('axis=1 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.mean(a, [1]);

    expect(res.dtype).toBe('float32');
    expect(res.shape).toEqual([3]);
    expectArraysClose(await res.data(), [1.5, 1.5, 0.5]);
  });

  it('axis = -1 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.mean(a, [-1]);

    expect(res.dtype).toBe('float32');
    expect(res.shape).toEqual([3]);
    expectArraysClose(await res.data(), [1.5, 1.5, 0.5]);
  });

  it('2D, axis=1 provided as number', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [2, 3]);
    const res = tf.mean(a, 1);

    expect(res.shape).toEqual([2]);
    expect(res.dtype).toBe('float32');
    expectArraysClose(await res.data(), [2, 1 / 3]);
  });

  it('axis=0,1 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.mean(a, [0, 1]);

    expect(res.shape).toEqual([]);
    expect(res.dtype).toBe('float32');
    expectArraysClose(await res.data(), [7 / 6]);
  });

  it('gradients', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const dy = tf.scalar(1.5);

    const da = tf.grad(a => a.mean())(a, dy);
    const dyVal = await dy.array();
    expect(da.shape).toEqual(a.shape);
    expectArraysClose(await da.data(), [
      dyVal / a.size, dyVal / a.size, dyVal / a.size, dyVal / a.size,
      dyVal / a.size, dyVal / a.size
    ]);
  });

  it('gradient with clones', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const dy = tf.scalar(1.5);

    const da = tf.grad(a => a.clone().mean().clone())(a, dy);
    const dyVal = await dy.array();
    expect(da.shape).toEqual(a.shape);
    expectArraysClose(await da.data(), [
      dyVal / a.size, dyVal / a.size, dyVal / a.size, dyVal / a.size,
      dyVal / a.size, dyVal / a.size
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

  it('accepts a tensor-like object', async () => {
    const r = tf.mean([[1, 2, 3], [0, 0, 1]]);

    expect(r.dtype).toBe('float32');
    expectArraysClose(await r.data(), 7 / 6);
  });

  it('throws error for string tensor', () => {
    expect(() => tf.mean(['a']))
        .toThrowError(/Argument 'x' passed to 'mean' must be numeric tensor/);
  });
});

describeWithFlags('norm', ALL_ENVS, () => {
  it('scalar norm', async () => {
    const a = tf.scalar(-22.0);
    const norm = tf.norm(a);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), 22);
  });

  it('vector inf norm', async () => {
    const a = tf.tensor1d([1, -2, 3, -4]);
    const norm = tf.norm(a, Infinity);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), 4);
  });

  it('vector -inf norm', async () => {
    const a = tf.tensor1d([1, -2, 3, -4]);
    const norm = tf.norm(a, -Infinity);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), 1);
  });

  it('vector 1 norm', async () => {
    const a = tf.tensor1d([1, -2, 3, -4]);
    const norm = tf.norm(a, 1);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), 10);
  });

  it('vector euclidean norm', async () => {
    const a = tf.tensor1d([1, -2, 3, -4]);
    const norm = tf.norm(a, 'euclidean');

    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), 5.4772);
  });

  it('vector 2-norm', async () => {
    const a = tf.tensor1d([1, -2, 3, -4]);
    const norm = tf.norm(a, 2);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), 5.4772);
  });

  it('vector >2-norm to throw error', () => {
    const a = tf.tensor1d([1, -2, 3, -4]);
    expect(() => tf.norm(a, 3)).toThrowError();
  });

  it('matrix inf norm', async () => {
    const a = tf.tensor2d([1, 2, -3, 1, 0, 1], [3, 2]);
    const norm = tf.norm(a, Infinity, [0, 1]);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), 4);
  });

  it('matrix -inf norm', async () => {
    const a = tf.tensor2d([1, 2, -3, 1, 0, 1], [3, 2]);
    const norm = tf.norm(a, -Infinity, [0, 1]);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), 1);
  });

  it('matrix 1 norm', async () => {
    const a = tf.tensor2d([1, 2, -3, 1, 1, 1], [3, 2]);
    const norm = tf.norm(a, 1, [0, 1]);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), 5);
  });

  it('matrix euclidean norm', async () => {
    const a = tf.tensor2d([1, 2, -3, 1, 1, 1], [3, 2]);
    const norm = tf.norm(a, 'euclidean', [0, 1]);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), 4.123);
  });

  it('matrix fro norm', async () => {
    const a = tf.tensor2d([1, 2, -3, 1, 1, 1], [3, 2]);
    const norm = tf.norm(a, 'fro', [0, 1]);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), 4.123);
  });

  it('matrix other norm to throw error', () => {
    const a = tf.tensor2d([1, 2, -3, 1, 1, 1], [3, 2]);
    expect(() => tf.norm(a, 2, [0, 1])).toThrowError();
  });

  it('propagates NaNs for norm', async () => {
    const a = tf.tensor2d([1, 2, 3, NaN, 0, 1], [3, 2]);
    const norm = tf.norm(a);

    expect(norm.dtype).toBe('float32');
    expectArraysEqual(await norm.data(), NaN);
  });

  it('axis=null in 2D array norm', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const norm = tf.norm(a, Infinity);

    expect(norm.shape).toEqual([]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [3]);
  });

  it('2D array norm with keep dim', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const norm = tf.norm(a, Infinity, null, true /* keepDims */);

    expect(norm.shape).toEqual([1, 1]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [3]);
  });

  it('axis=0 in 2D array norm', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const norm = tf.norm(a, Infinity, [0]);

    expect(norm.shape).toEqual([2]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [3, 2]);
  });

  it('axis=1 in 2D array norm', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const norm = tf.norm(a, Infinity, [1]);

    expect(norm.dtype).toBe('float32');
    expect(norm.shape).toEqual([3]);
    expectArraysClose(await norm.data(), [2, 3, 1]);
  });

  it('axis=1 keepDims in 2D array norm', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const norm = tf.norm(a, Infinity, [1], true);

    expect(norm.dtype).toBe('float32');
    expect(norm.shape).toEqual([3, 1]);
    expectArraysClose(await norm.data(), [2, 3, 1]);
  });

  it('2D norm with axis=1 provided as number', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [2, 3]);
    const norm = tf.norm(a, Infinity, 1);

    expect(norm.shape).toEqual([2]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [3, 1]);
  });

  it('axis=0,1 in 2D array norm', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const norm = tf.norm(a, Infinity, [0, 1]);

    expect(norm.shape).toEqual([]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [3]);
  });

  it('axis=0,1 keepDims in 2D array norm', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const norm = tf.norm(a, Infinity, [0, 1], true);

    expect(norm.shape).toEqual([1, 1]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [3]);
  });

  it('3D norm axis=0,1, matrix inf norm', async () => {
    const a = tf.tensor3d([1, 2, -3, 1, 0, 1], [3, 2, 1]);
    const norm = tf.norm(a, Infinity, [0, 1]);

    expect(norm.shape).toEqual([1]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [4]);
  });

  it('axis=0,1 keepDims in 3D array norm', async () => {
    const a = tf.tensor3d([1, 2, 3, 0, 0, 1], [3, 2, 1]);
    const norm = tf.norm(a, Infinity, [0, 1], true);

    expect(norm.shape).toEqual([1, 1, 1]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [3]);
  });

  it('axis=0,1 keepDims in 3D array norm', async () => {
    const a = tf.tensor3d([1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 1], [3, 2, 2]);
    const norm = tf.norm(a, Infinity, [0, 1], true);

    expect(norm.shape).toEqual([1, 1, 2]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [4, 3]);
  });

  it('axis=null in 3D array norm', async () => {
    const a = tf.tensor3d([1, 2, 3, 0, 0, 1], [3, 2, 1]);
    const norm = tf.norm(a, Infinity);

    expect(norm.shape).toEqual([]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [3]);
  });

  it('axis=null in 4D array norm', async () => {
    const a = tf.tensor4d([1, 2, 3, 0, 0, 1], [3, 2, 1, 1]);
    const norm = tf.norm(a, Infinity);

    expect(norm.shape).toEqual([]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [3]);
  });

  it('axis=0,1 in 4D array norm', async () => {
    const a = tf.tensor4d(
        [
          1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 1,
          1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 1
        ],
        [3, 2, 2, 2]);
    const norm = tf.norm(a, Infinity, [0, 1]);

    expect(norm.shape).toEqual([2, 2]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [4, 3, 4, 3]);
  });

  it('axis=0,1 in 4D array norm', async () => {
    const a = tf.tensor4d(
        [
          1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 1,
          1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 0, 1
        ],
        [3, 2, 2, 2]);
    const norm = tf.norm(a, Infinity, [0, 1], true);

    expect(norm.shape).toEqual([1, 1, 2, 2]);
    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), [4, 3, 4, 3]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.norm({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'norm' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const norm = tf.norm([1, -2, 3, -4], 1);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(await norm.data(), 10);
  });

  it('throws error for string tensors', () => {
    expect(() => tf.norm([
      'a', 'b'
    ])).toThrowError(/Argument 'x' passed to 'norm' must be numeric tensor/);
  });
});
