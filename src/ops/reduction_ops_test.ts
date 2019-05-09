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
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {expectArraysClose, expectArraysEqual} from '../test_util';

import * as reduce_util from './reduce_util';

describeWithFlags('Reduction: min', ALL_ENVS, () => {
  it('Tensor1D', async () => {
    const a = tf.tensor1d([3, -1, 0, 100, -7, 2]);
    expectArraysClose(await tf.min(a).data(), -7);
  });

  it('ignores NaNs', async () => {
    const a = tf.tensor1d([3, NaN, 2]);
    expectArraysEqual(await tf.min(a).data(), 2);
  });

  it('2D', async () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    expectArraysClose(await tf.min(a).data(), -7);
  });

  it('2D axis=[0,1]', async () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    expectArraysClose(await tf.min(a, [0, 1]).data(), -7);
  });

  it('2D, axis=0', async () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    const r = tf.min(a, 0);

    expect(r.shape).toEqual([3]);
    expectArraysClose(await r.data(), [3, -7, 0]);
  });

  it('2D, axis=0, keepDims', async () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    const r = tf.min(a, 0, true /* keepDims */);

    expect(r.shape).toEqual([1, 3]);
    expectArraysClose(await r.data(), [3, -7, 0]);
  });

  it('2D, axis=1 provided as a number', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const r = tf.min(a, 1);
    expectArraysClose(await r.data(), [2, -7]);
  });

  it('2D, axis = -1 provided as a number', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const r = tf.min(a, -1);
    expectArraysClose(await r.data(), [2, -7]);
  });

  it('2D, axis=[1]', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const r = tf.min(a, [1]);
    expectArraysClose(await r.data(), [2, -7]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.min({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'min' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    expectArraysClose(await tf.min([3, -1, 0, 100, -7, 2]).data(), -7);
  });

  it('min gradient: Scalar', async () => {
    const x = tf.scalar(42);
    const dy = tf.scalar(-1);
    const gradients = tf.grad(v => tf.min(v))(x, dy);
    expectArraysClose(await gradients.data(), -1);
  });

  it('gradient with clones', async () => {
    const x = tf.scalar(42);
    const dy = tf.scalar(-1);
    const gradients = tf.grad(v => tf.min(v.clone()).clone())(x, dy);
    expectArraysClose(await gradients.data(), -1);
  });

  it('min gradient: 1D, ties', async () => {
    const x = tf.tensor1d([-1, -3, -7, -7]);
    const dy = tf.scalar(-1);
    const gradients = tf.grad(v => tf.min(v))(x, dy);
    expectArraysClose(await gradients.data(), [0, 0, -1, -1]);
  });

  it('min gradient: 2D, axes=-1, keepDims=false', async () => {
    const x = tf.tensor2d([[-0, -20, -10], [10, 30, 20]]);
    const dy = tf.tensor1d([-1, -1]);
    const axis = -1;
    const gradients = tf.grad(v => tf.min(v, axis))(x, dy);
    expectArraysClose(await gradients.data(), [0, -1, 0, -1, 0, 0]);
    expect(gradients.shape).toEqual([2, 3]);
  });

  it('min gradient: ties, 2D, axes=-1, keepDims=false', async () => {
    const x = tf.tensor2d([[0, -20, -20], [10, 30, 10]]);
    const dy = tf.tensor1d([-1, -1]);
    const axis = -1;
    const gradients = tf.grad(v => tf.min(v, axis))(x, dy);
    expectArraysClose(await gradients.data(), [0, -1, -1, -1, 0, -1]);
    expect(gradients.shape).toEqual([2, 3]);
  });

  it('min gradient: 2D, axes=0, keepDims=false', async () => {
    const x = tf.tensor2d([[0, 20, 10], [-10, -30, 20]]);
    const dy = tf.tensor1d([-1, -1, -1]);
    const axis = 0;
    const gradients = tf.grad(v => tf.max(v, axis))(x, dy);
    expectArraysClose(await gradients.data(), [-1, -1, 0, 0, 0, -1]);
    expect(gradients.shape).toEqual([2, 3]);
  });

  it('min gradient: 2D, axes=-1, keepDims=true', async () => {
    const x = tf.tensor2d([[0, -20, -10], [10, 30, 20]]);
    const dy = tf.tensor2d([[-1], [-1]]);
    const axis = -1;
    const keepDims = true;
    const gradients = tf.grad(v => tf.min(v, axis, keepDims))(x, dy);
    expectArraysClose(await gradients.data(), [0, -1, 0, -1, 0, 0]);
    expect(gradients.shape).toEqual([2, 3]);
  });

  it('min gradient: 2D, axes=0, keepDims=true', async () => {
    const x = tf.tensor2d([[0, -20, -10], [10, 30, -20]]);
    const dy = tf.tensor2d([[-1, -1, -1]]);
    const axis = 0;
    const keepDims = true;
    const gradients = tf.grad(v => tf.min(v, axis, keepDims))(x, dy);
    expectArraysClose(await gradients.data(), [-1, -1, 0, 0, 0, -1]);
    expect(gradients.shape).toEqual([2, 3]);
  });

  it('min gradient: 3D, axes=[1, 2], keepDims=false', async () => {
    const x = tf.tensor3d([[[0, -20], [-10, -15]], [[10, 30], [20, 15]]]);
    const dy = tf.tensor1d([-1, -1]);
    const axis = [1, 2];
    const gradients = tf.grad(v => tf.min(v, axis))(x, dy);
    expectArraysClose(await gradients.data(), [0, -1, 0, 0, -1, 0, 0, 0]);
    expect(gradients.shape).toEqual([2, 2, 2]);
  });

  it('min gradient: ties, 3D, axes=[1, 2], keepDims=false', async () => {
    const x = tf.tensor3d([[[0, -20], [-20, -20]], [[10, 30], [10, 15]]]);
    const dy = tf.tensor1d([-1, -1]);
    const axis = [1, 2];
    const gradients = tf.grad(v => tf.min(v, axis))(x, dy);
    expectArraysClose(await gradients.data(), [0, -1, -1, -1, -1, 0, -1, 0]);
    expect(gradients.shape).toEqual([2, 2, 2]);
  });

  it('min gradient: 3D, axes=2, keepDims=false', async () => {
    const x = tf.tensor3d([[[0, -20], [-10, -15]], [[10, 30], [20, 15]]]);
    const dy = tf.tensor2d([[-1, -1], [-1, -1]]);
    const axis = 2;
    const gradients = tf.grad(v => tf.min(v, axis))(x, dy);
    expectArraysClose(await gradients.data(), [0, -1, 0, -1, -1, 0, 0, -1]);
    expect(gradients.shape).toEqual([2, 2, 2]);
  });

  it('min gradient: 3D, axes=2, keepDims=true', async () => {
    const x = tf.tensor3d([[[0, -20], [-10, -15]], [[10, 30], [20, 15]]]);
    const dy = tf.tensor3d([[[-1], [-1]], [[-1], [-1]]]);
    const axis = 2;
    const keepDims = true;
    const gradients = tf.grad(v => tf.min(v, axis, keepDims))(x, dy);
    expectArraysClose(await gradients.data(), [0, -1, 0, -1, -1, 0, 0, -1]);
    expect(gradients.shape).toEqual([2, 2, 2]);
  });

  it('min gradient: ties, 4D, axes=[1, 2, 3], keepDims=false', async () => {
    const x = tf.tensor4d([
      [[[0, -20], [-20, -20]], [[10, 30], [10, 30]]],
      [[[0, 20], [20, 20]], [[-10, -30], [-10, -30]]]
    ]);
    const dy = tf.tensor1d([-1, -1]);
    const axis = [1, 2, 3];
    const gradients = tf.grad(v => tf.min(v, axis))(x, dy);
    expectArraysClose(
        await gradients.data(),
        [0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1]);
    expect(gradients.shape).toEqual([2, 2, 2, 2]);
  });

  it('min gradient: ties, 4D, axes=[2, 3], keepDims=true', async () => {
    const x = tf.tensor4d([
      [[[0, -20], [-20, -20]], [[10, 30], [10, 30]]],
      [[[0, 20], [20, 20]], [[-10, -30], [-10, -30]]]
    ]);
    const dy = tf.tensor4d([[[[-1]], [[-2]]], [[[-3]], [[-4]]]]);
    const axis = [2, 3];
    const keepDims = true;
    const gradients = tf.grad(v => tf.min(v, axis, keepDims))(x, dy);
    expectArraysClose(
        await gradients.data(),
        [0, -1, -1, -1, -2, 0, -2, 0, -3, 0, 0, 0, 0, -4, 0, -4]);
    expect(gradients.shape).toEqual([2, 2, 2, 2]);
  });

  it('throws error for string tensor', () => {
    expect(() => tf.min(['a']))
        .toThrowError(/Argument 'x' passed to 'min' must be numeric tensor/);
  });
});

describeWithFlags('Reduction: max', ALL_ENVS, () => {
  it('with one element dominating', async () => {
    const a = tf.tensor1d([3, -1, 0, 100, -7, 2]);
    const r = tf.max(a);
    expectArraysClose(await r.data(), 100);
  });

  it('with all elements being the same', async () => {
    const a = tf.tensor1d([3, 3, 3]);
    const r = tf.max(a);
    expectArraysClose(await r.data(), 3);
  });

  it('ignores NaNs', async () => {
    expectArraysClose(await tf.max([3, NaN, 2]).data(), 3);
  });

  it('2D', async () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    expectArraysClose(await tf.max(a).data(), 100);
  });

  it('2D axis=[0,1]', async () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    expectArraysClose(await tf.max(a, [0, 1]).data(), 100);
  });

  it('2D, axis=0', async () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    const r = tf.max(a, [0]);

    expect(r.shape).toEqual([3]);
    expectArraysClose(await r.data(), [100, -1, 2]);
  });

  it('2D, axis=0, keepDims', async () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    const r = tf.max(a, [0], true /* keepDims */);

    expect(r.shape).toEqual([1, 3]);
    expectArraysClose(await r.data(), [100, -1, 2]);
  });

  it('2D, axis=1 provided as a number', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const r = tf.max(a, 1);
    expectArraysClose(await r.data(), [5, 100]);
  });

  it('2D, axis = -1 provided as a number', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const r = tf.max(a, -1);
    expectArraysClose(await r.data(), [5, 100]);
  });

  it('2D, axis=[1]', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const r = tf.max(a, [1]);
    expectArraysClose(await r.data(), [5, 100]);
  });

  it('6D, axis=[5]', async () => {
    const a = tf.range(0, 64).reshape([2, 2, 2, 2, 2, 2]);
    const r = tf.max(a, [5]);
    const expectedResult = [
      1,  3,  5,  7,  9,  11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
      33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63
    ];
    expectArraysClose(await r.data(), expectedResult);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.max({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'max' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const r = tf.max([3, -1, 0, 100, -7, 2]);
    expectArraysClose(await r.data(), 100);
  });

  it('max gradient: Scalar', async () => {
    const x = tf.scalar(42);
    const dy = tf.scalar(-1);
    const gradients = tf.grad(v => tf.max(v))(x, dy);
    expectArraysClose(await gradients.data(), [-1]);
  });

  it('gradient with clones', async () => {
    const x = tf.scalar(42);
    const dy = tf.scalar(-1);
    const gradients = tf.grad(v => tf.max(v.clone()).clone())(x, dy);
    expectArraysClose(await gradients.data(), [-1]);
  });

  it('max gradient: 1D, ties', async () => {
    const x = tf.tensor1d([1, 3, 7, 7]);
    const dy = tf.scalar(-1);
    const gradients = tf.grad(v => tf.max(v))(x, dy);
    expectArraysClose(await gradients.data(), [0, 0, -1, -1]);
  });

  it('max gradient: 2D, axes=-1, keepDims=false', async () => {
    const x = tf.tensor2d([[0, 20, 10], [-10, -30, -20]]);
    const dy = tf.tensor1d([-1, -1]);
    const axis = -1;
    const gradients = tf.grad(v => tf.max(v, axis))(x, dy);
    expectArraysClose(await gradients.data(), [0, -1, 0, -1, 0, 0]);
    expect(gradients.shape).toEqual([2, 3]);
  });

  it('max gradient: ties, 2D, axes=-1, keepDims=false', async () => {
    const x = tf.tensor2d([[0, 20, 20], [-10, -30, -10]]);
    const dy = tf.tensor1d([-1, -1]);
    const axis = -1;
    const gradients = tf.grad(v => tf.max(v, axis))(x, dy);
    expectArraysClose(await gradients.data(), [0, -1, -1, -1, 0, -1]);
    expect(gradients.shape).toEqual([2, 3]);
  });

  it('max gradient: 2D, axes=0, keepDims=false', async () => {
    const x = tf.tensor2d([[0, 20, 10], [-10, -30, 20]]);
    const dy = tf.tensor1d([-1, -1, -1]);
    const axis = 0;
    const gradients = tf.grad(v => tf.max(v, axis))(x, dy);
    expectArraysClose(await gradients.data(), [-1, -1, 0, 0, 0, -1]);
    expect(gradients.shape).toEqual([2, 3]);
  });

  it('max gradient: 2D, axes=-1, keepDims=true', async () => {
    const x = tf.tensor2d([[0, 20, 10], [-10, -30, -20]]);
    const dy = tf.tensor2d([[-1], [-1]]);
    const axis = -1;
    const keepDims = true;
    const gradients = tf.grad(v => tf.max(v, axis, keepDims))(x, dy);
    expectArraysClose(await gradients.data(), [0, -1, 0, -1, 0, 0]);
    expect(gradients.shape).toEqual([2, 3]);
  });

  it('max gradient: 2D, axes=0, keepDims=true', async () => {
    const x = tf.tensor2d([[0, 20, 10], [-10, -30, 20]]);
    const dy = tf.tensor2d([[-1, -1, -1]]);
    const axis = 0;
    const keepDims = true;
    const gradients = tf.grad(v => tf.max(v, axis, keepDims))(x, dy);
    expectArraysClose(await gradients.data(), [-1, -1, 0, 0, 0, -1]);
    expect(gradients.shape).toEqual([2, 3]);
  });

  it('max gradient: 3D, axes=[1, 2], keepDims=false', async () => {
    const x = tf.tensor3d([[[0, 20], [10, 15]], [[-10, -30], [-20, -15]]]);
    const dy = tf.tensor1d([-1, -1]);
    const axis = [1, 2];
    const gradients = tf.grad(v => tf.max(v, axis))(x, dy);
    expectArraysClose(await gradients.data(), [0, -1, 0, 0, -1, 0, 0, 0]);
    expect(gradients.shape).toEqual([2, 2, 2]);
  });

  it('max gradient: ties, 3D, axes=[1, 2], keepDims=false', async () => {
    const x = tf.tensor3d([[[0, 20], [20, 20]], [[-10, -30], [-10, -15]]]);
    const dy = tf.tensor1d([-1, -1]);
    const axis = [1, 2];
    const gradients = tf.grad(v => tf.max(v, axis))(x, dy);
    expectArraysClose(await gradients.data(), [0, -1, -1, -1, -1, 0, -1, 0]);
    expect(gradients.shape).toEqual([2, 2, 2]);
  });

  it('max gradient: 3D, axes=2, keepDims=false', async () => {
    const x = tf.tensor3d([[[0, 20], [10, 15]], [[-10, -30], [-20, -15]]]);
    const dy = tf.tensor2d([[-1, -1], [-1, -1]]);
    const axis = 2;
    const gradients = tf.grad(v => tf.max(v, axis))(x, dy);
    expectArraysClose(await gradients.data(), [0, -1, 0, -1, -1, 0, 0, -1]);
    expect(gradients.shape).toEqual([2, 2, 2]);
  });

  it('max gradient: 3D, axes=2, keepDims=true', async () => {
    const x = tf.tensor3d([[[0, 20], [10, 15]], [[-10, -30], [-20, -15]]]);
    const dy = tf.tensor3d([[[-1], [-1]], [[-1], [-1]]]);
    const axis = 2;
    const keepDims = true;
    const gradients = tf.grad(v => tf.max(v, axis, keepDims))(x, dy);
    expectArraysClose(await gradients.data(), [0, -1, 0, -1, -1, 0, 0, -1]);
    expect(gradients.shape).toEqual([2, 2, 2]);
  });

  it('max gradient: ties, 4D, axes=[1, 2, 3], keepDims=false', async () => {
    const x = tf.tensor4d([
      [[[0, 20], [20, 20]], [[-10, -30], [-10, -30]]],
      [[[0, -20], [-20, -20]], [[10, 30], [10, 30]]]
    ]);
    const dy = tf.tensor1d([-1, -1]);
    const axis = [1, 2, 3];
    const gradients = tf.grad(v => tf.max(v, axis))(x, dy);
    expectArraysClose(
        await gradients.data(),
        [0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1]);
    expect(gradients.shape).toEqual([2, 2, 2, 2]);
  });

  it('max gradient: ties, 4D, axes=[2, 3], keepDims=true', async () => {
    const x = tf.tensor4d([
      [[[0, 20], [20, 20]], [[-10, -30], [-10, -30]]],
      [[[0, -20], [-20, -20]], [[10, 30], [10, 30]]]
    ]);
    const dy = tf.tensor4d([[[[-1]], [[-2]]], [[[-3]], [[-4]]]]);
    const axis = [2, 3];
    const keepDims = true;
    const gradients = tf.grad(v => tf.max(v, axis, keepDims))(x, dy);
    expectArraysClose(
        await gradients.data(),
        [0, -1, -1, -1, -2, 0, -2, 0, -3, 0, 0, 0, 0, -4, 0, -4]);
    expect(gradients.shape).toEqual([2, 2, 2, 2]);
  });

  it('throws error for string tensor', () => {
    expect(() => tf.max(['a']))
        .toThrowError(/Argument 'x' passed to 'max' must be numeric tensor/);
  });
});

describeWithFlags('Reduction: argmax', ALL_ENVS, () => {
  it('Tensor1D', async () => {
    const a = tf.tensor1d([1, 0, 3, 2]);
    const result = tf.argMax(a);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), 2);
  });

  it('one value', async () => {
    const a = tf.tensor1d([10]);
    const result = tf.argMax(a);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), 0);
  });

  it('N > than parallelization threshold', async () => {
    const n = reduce_util.PARALLELIZE_THRESHOLD * 2;
    const values = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      values[i] = i;
    }
    const a = tf.tensor1d(values);
    const result = tf.argMax(a);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), n - 1);
  });

  it('3D, N > than parallelization threshold', async () => {
    const n = reduce_util.PARALLELIZE_THRESHOLD * 2;
    const values = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      values[i] = i;
    }
    const a = tf.tensor3d(values, [1, 1, n]);
    const result = tf.argMax(a, -1);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), n - 1);
  });

  it('max index corresponds to start of a non-initial window', async () => {
    const n = reduce_util.PARALLELIZE_THRESHOLD * 2;
    const windowSize = reduce_util.computeOptimalWindowSize(n);
    const values = new Float32Array(n);
    const index = windowSize * 2;
    values[index] = 1;
    const a = tf.tensor1d(values);
    const result = tf.argMax(a);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), index);
  });

  it('5D, max index corresponds to start of a non-initial window', async () => {
    const n = reduce_util.PARALLELIZE_THRESHOLD * 2;
    const windowSize = reduce_util.computeOptimalWindowSize(n);
    const values = new Float32Array(n);
    const index = windowSize * 2;
    values[index] = 1;
    const a = tf.tensor5d(values, [1, 1, 1, 1, n]);
    const result = tf.argMax(a, -1);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), index);
  });

  it('ignores NaNs', async () => {
    const a = tf.tensor1d([0, 3, 5, NaN, 3]);
    const res = tf.argMax(a);
    expect(res.dtype).toBe('int32');
    expectArraysEqual(await res.data(), 2);
  });

  it('2D, no axis specified', async () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    expectArraysEqual(await tf.argMax(a).data(), [1, 0, 1]);
  });

  it('4D, no axis specified', async () => {
    const a = tf.tensor4d([3, -1, 0, 100, -7, 2], [2, 1, 1, 3]);
    expectArraysEqual(await tf.argMax(a).data(), [1, 0, 1]);
  });

  it('2D, axis=0', async () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    const r = tf.argMax(a, 0);

    expect(r.shape).toEqual([3]);
    expect(r.dtype).toBe('int32');
    expectArraysEqual(await r.data(), [1, 0, 1]);
  });

  it('6D, axis=0', async () => {
    const a = tf.tensor6d([3, -1, 0, 100, -7, 2], [2, 1, 1, 1, 1, 3]);
    const r = tf.argMax(a, 0);

    expect(r.shape).toEqual([1, 1, 1, 1, 3]);
    expect(r.dtype).toBe('int32');
    expectArraysEqual(await r.data(), [1, 0, 1]);
  });

  it('2D, axis=1', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const r = tf.argMax(a, 1);
    expect(r.dtype).toBe('int32');
    expectArraysEqual(await r.data(), [2, 0]);
  });

  it('2D, axis = -1', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const r = tf.argMax(a, -1);
    expect(r.dtype).toBe('int32');
    expectArraysEqual(await r.data(), [2, 0]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.argMax({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'argMax' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.argMax([1, 0, 3, 2]);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), 2);
  });

  it('accepts tensor with bool values', async () => {
    const t = tf.tensor1d([0, 1], 'bool');
    const result = tf.argMax(t);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), 1);
  });

  it('has gradient', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const dy = tf.ones([3], 'float32') as tf.Tensor1D;
    const da = tf.grad((x: tf.Tensor2D) => tf.argMax(x))(a, dy);

    expect(da.dtype).toBe('float32');
    expect(da.shape).toEqual([2, 3]);
    expectArraysClose(await da.data(), [0, 0, 0, 0, 0, 0]);
  });

  it('gradient with clones', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const dy = tf.ones([3], 'float32') as tf.Tensor1D;
    const da = tf.grad((x: tf.Tensor2D) => tf.argMax(x.clone()).clone())(a, dy);

    expect(da.dtype).toBe('float32');
    expect(da.shape).toEqual([2, 3]);
    expectArraysClose(await da.data(), [0, 0, 0, 0, 0, 0]);
  });

  it('throws error for string tensor', () => {
    expect(() => tf.argMax(['a']))
        .toThrowError(/Argument 'x' passed to 'argMax' must be numeric tensor/);
  });
});

describeWithFlags('Reduction: argmin', ALL_ENVS, () => {
  it('Tensor1D', async () => {
    const a = tf.tensor1d([1, 0, 3, 2]);
    const result = tf.argMin(a);
    expectArraysEqual(await result.data(), 1);
  });

  it('one value', async () => {
    const a = tf.tensor1d([10]);
    const result = tf.argMin(a);
    expectArraysEqual(await result.data(), 0);
  });

  it('N > than parallelization threshold', async () => {
    const n = reduce_util.PARALLELIZE_THRESHOLD * 2;
    const values = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      values[i] = n - i;
    }
    const a = tf.tensor1d(values);
    const result = tf.argMin(a);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), n - 1);
  });

  it('4D, N > than parallelization threshold', async () => {
    const n = reduce_util.PARALLELIZE_THRESHOLD * 2;
    const values = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      values[i] = n - i;
    }
    const a = tf.tensor4d(values, [1, 1, 1, n]);
    const result = tf.argMin(a, -1);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), n - 1);
  });

  it('min index corresponds to start of a non-initial window', async () => {
    const n = reduce_util.PARALLELIZE_THRESHOLD * 2;
    const windowSize = reduce_util.computeOptimalWindowSize(n);
    const values = new Float32Array(n);
    const index = windowSize * 2;
    values[index] = -1;
    const a = tf.tensor1d(values);
    const result = tf.argMin(a);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), index);
  });

  it('ignores NaNs', async () => {
    const a = tf.tensor1d([5, 0, NaN, -1, 3]);
    const res = tf.argMin(a);
    expectArraysEqual(await res.data(), 3);
  });

  it('3D, ignores NaNs', async () => {
    const a = tf.tensor3d([5, 0, NaN, -1, 3], [1, 1, 5]);
    const res = tf.argMin(a, -1);
    expectArraysEqual(await res.data(), 3);
  });

  it('2D, no axis specified', async () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    expectArraysEqual(await tf.argMin(a).data(), [0, 1, 0]);
  });

  it('2D, axis=0', async () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    const r = tf.argMin(a, 0);

    expect(r.shape).toEqual([3]);
    expect(r.dtype).toBe('int32');
    expectArraysEqual(await r.data(), [0, 1, 0]);
  });

  it('2D, axis=1', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, -8], [2, 3]);
    const r = tf.argMin(a, 1);
    expectArraysEqual(await r.data(), [1, 2]);
  });

  it('2D, axis = -1', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, -8], [2, 3]);
    const r = tf.argMin(a, -1);
    expectArraysEqual(await r.data(), [1, 2]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.argMin({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'argMin' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.argMin([1, 0, 3, 2]);
    expectArraysEqual(await result.data(), 1);
  });

  it('accepts tensor with bool values', async () => {
    const t = tf.tensor1d([0, 1], 'bool');
    const result = tf.argMin(t);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(await result.data(), 0);
  });

  it('has gradient', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const dy = tf.ones([3], 'float32') as tf.Tensor1D;
    const da = tf.grad((x: tf.Tensor2D) => tf.argMin(x))(a, dy);

    expect(da.dtype).toBe('float32');
    expect(da.shape).toEqual([2, 3]);
    expectArraysClose(await da.data(), [0, 0, 0, 0, 0, 0]);
  });

  it('gradient with clones', async () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const dy = tf.ones([3], 'float32') as tf.Tensor1D;
    const da = tf.grad((x: tf.Tensor2D) => tf.argMin(x.clone()).clone())(a, dy);

    expect(da.dtype).toBe('float32');
    expect(da.shape).toEqual([2, 3]);
    expectArraysClose(await da.data(), [0, 0, 0, 0, 0, 0]);
  });

  it('throws error for string tensor', () => {
    expect(() => tf.argMin(['a']))
        .toThrowError(/Argument 'x' passed to 'argMin' must be numeric tensor/);
  });
});

describeWithFlags('Reduction: logSumExp', ALL_ENVS, () => {
  it('0', async () => {
    const a = tf.scalar(0);
    const result = tf.logSumExp(a);
    expectArraysClose(await result.data(), 0);
  });

  it('basic', async () => {
    const a = tf.tensor1d([1, 2, -3]);
    const result = tf.logSumExp(a);

    expectArraysClose(
        await result.data(),
        Math.log(Math.exp(1) + Math.exp(2) + Math.exp(-3)));
  });

  it('propagates NaNs', async () => {
    const a = tf.tensor1d([1, 2, NaN]);
    const result = tf.logSumExp(a);
    expectArraysEqual(await result.data(), NaN);
  });

  it('axes=0 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const r = tf.logSumExp(a, [0]);

    expect(r.shape).toEqual([2]);
    const expected = [
      Math.log(Math.exp(1) + Math.exp(3) + Math.exp(0)),
      Math.log(Math.exp(2) + Math.exp(0) + Math.exp(1))
    ];
    expectArraysClose(await r.data(), expected);
  });

  it('axes=0 in 2D array, keepDims', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const r = tf.logSumExp(a, [0], true /* keepDims */);

    expect(r.shape).toEqual([1, 2]);
    const expected = [
      Math.log(Math.exp(1) + Math.exp(3) + Math.exp(0)),
      Math.log(Math.exp(2) + Math.exp(0) + Math.exp(1))
    ];
    expectArraysClose(await r.data(), expected);
  });

  it('axes=1 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.logSumExp(a, [1]);

    expect(res.shape).toEqual([3]);
    const expected = [
      Math.log(Math.exp(1) + Math.exp(2)),
      Math.log(Math.exp(3) + Math.exp(0)),
      Math.log(Math.exp(0) + Math.exp(1)),
    ];
    expectArraysClose(await res.data(), expected);
  });

  it('axes = -1 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.logSumExp(a, -1);

    expect(res.shape).toEqual([3]);
    const expected = [
      Math.log(Math.exp(1) + Math.exp(2)),
      Math.log(Math.exp(3) + Math.exp(0)),
      Math.log(Math.exp(0) + Math.exp(1)),
    ];
    expectArraysClose(await res.data(), expected);
  });

  it('2D, axes=1 provided as a single digit', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [2, 3]);
    const res = tf.logSumExp(a, 1);

    expect(res.shape).toEqual([2]);
    const expected = [
      Math.log(Math.exp(1) + Math.exp(2) + Math.exp(3)),
      Math.log(Math.exp(0) + Math.exp(0) + Math.exp(1))
    ];
    expectArraysClose(await res.data(), expected);
  });

  it('axes=0,1 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.logSumExp(a, [0, 1]);

    expect(res.shape).toEqual([]);
    const expected = [Math.log(
        Math.exp(1) + Math.exp(2) + Math.exp(3) + Math.exp(0) + Math.exp(0) +
        Math.exp(1))];
    expectArraysClose(await res.data(), expected);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.logSumExp({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'logSumExp' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.logSumExp([1, 2, -3]);
    expectArraysClose(
        await result.data(),
        Math.log(Math.exp(1) + Math.exp(2) + Math.exp(-3)));
  });

  it('throws error for string tensor', () => {
    expect(() => tf.logSumExp(['a']))
        .toThrowError(
            /Argument 'x' passed to 'logSumExp' must be numeric tensor/);
  });
});

describeWithFlags('Reduction: sum', ALL_ENVS, () => {
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

describeWithFlags('Reduction: prod', ALL_ENVS, () => {
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

describeWithFlags('Reduction: mean', ALL_ENVS, () => {
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

describeWithFlags('Reduction: moments', ALL_ENVS, () => {
  it('basic', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const {mean, variance} = tf.moments(a);

    expect(mean.dtype).toBe('float32');
    expect(variance.dtype).toBe('float32');
    expectArraysClose(await mean.data(), 7 / 6);
    expectArraysClose(await variance.data(), 1.1389);
  });

  it('propagates NaNs', async () => {
    const a = tf.tensor2d([1, 2, 3, NaN, 0, 1], [3, 2]);
    const {mean, variance} = tf.moments(a);

    expect(mean.dtype).toBe('float32');
    expect(variance.dtype).toBe('float32');
    expectArraysEqual(await mean.data(), NaN);
    expectArraysEqual(await variance.data(), NaN);
  });

  it('moments(int32) => float32', async () => {
    const a = tf.tensor1d([1, 5, 7, 3], 'int32');
    const {mean, variance} = tf.moments(a);

    expect(mean.dtype).toBe('float32');
    expect(variance.dtype).toBe('float32');
    expectArraysClose(await mean.data(), 4);
    expectArraysClose(await variance.data(), 5);
  });

  it('moments(bool) => float32', async () => {
    const a = tf.tensor1d([true, false, false, true, true], 'bool');
    const {mean, variance} = tf.moments(a);

    expect(mean.dtype).toBe('float32');
    expect(variance.dtype).toBe('float32');
    expectArraysClose(await mean.data(), 3 / 5);
    expectArraysClose(await variance.data(), 0.23999998);
  });

  it('2D array with keep dim', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const {mean, variance} = tf.moments(a, null, true /* keepDims */);

    expect(mean.shape).toEqual([1, 1]);
    expect(mean.dtype).toBe('float32');
    expect(variance.shape).toEqual([1, 1]);
    expect(variance.dtype).toBe('float32');
    expectArraysClose(await mean.data(), [7 / 6]);
    expectArraysClose(await variance.data(), [1.138889]);
  });

  it('axis=0 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const {mean, variance} = tf.moments(a, [0]);

    expect(mean.shape).toEqual([2]);
    expect(mean.dtype).toBe('float32');
    expect(variance.shape).toEqual([2]);
    expect(variance.dtype).toBe('float32');
    expectArraysClose(await mean.data(), [4 / 3, 1]);
    expectArraysClose(await variance.data(), [1.556, 2 / 3]);
  });

  it('axis=1 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const {mean, variance} = tf.moments(a, [1]);

    expect(mean.dtype).toBe('float32');
    expect(mean.shape).toEqual([3]);
    expect(variance.dtype).toBe('float32');
    expect(variance.shape).toEqual([3]);
    expectArraysClose(await mean.data(), [1.5, 1.5, 0.5]);
    expectArraysClose(await variance.data(), [0.25, 2.25, 0.25]);
  });

  it('2D, axis=1 provided as number', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [2, 3]);
    const {mean, variance} = tf.moments(a, 1);

    expect(mean.shape).toEqual([2]);
    expect(mean.dtype).toBe('float32');
    expect(variance.shape).toEqual([2]);
    expect(variance.dtype).toBe('float32');
    expectArraysClose(await mean.data(), [2, 1 / 3]);
    expectArraysClose(await variance.data(), [2 / 3, 0.222]);
  });

  it('2D, axis=-1 provided as number', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [2, 3]);
    const {mean, variance} = tf.moments(a, -1);

    expect(mean.shape).toEqual([2]);
    expect(mean.dtype).toBe('float32');
    expect(variance.shape).toEqual([2]);
    expect(variance.dtype).toBe('float32');
    expectArraysClose(await mean.data(), [2, 1 / 3]);
    expectArraysClose(await variance.data(), [2 / 3, 0.222]);
  });

  it('axis=0,1 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const {mean, variance} = tf.moments(a, [0, 1]);

    expect(mean.shape).toEqual([]);
    expect(mean.dtype).toBe('float32');
    expect(variance.shape).toEqual([]);
    expect(variance.dtype).toBe('float32');
    expectArraysClose(await mean.data(), [7 / 6]);
    expectArraysClose(await variance.data(), [1.1389]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.moments({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'moments' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const {mean, variance} = tf.moments([1, 2, 3, 0, 0, 1]);

    expect(mean.dtype).toBe('float32');
    expect(variance.dtype).toBe('float32');
    expectArraysClose(await mean.data(), 7 / 6);
    expectArraysClose(await variance.data(), 1.1389);
  });
});

describeWithFlags('Reduction: norm', ALL_ENVS, () => {
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

describeWithFlags('Reduction: all', ALL_ENVS, () => {
  it('Tensor1D', async () => {
    let a = tf.tensor1d([0, 0, 0], 'bool');
    expectArraysClose(await tf.all(a).data(), 0);

    a = tf.tensor1d([1, 0, 1], 'bool');
    expectArraysClose(await tf.all(a).data(), 0);

    a = tf.tensor1d([1, 1, 1], 'bool');
    expectArraysClose(await tf.all(a).data(), 1);
  });

  it('ignores NaNs', async () => {
    const a = tf.tensor1d([1, NaN, 1], 'bool');
    expectArraysEqual(await tf.all(a).data(), 1);
  });

  it('2D', async () => {
    const a = tf.tensor2d([1, 1, 0, 0], [2, 2], 'bool');
    expectArraysClose(await tf.all(a).data(), 0);
  });

  it('2D axis=[0,1]', async () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    expectArraysClose(await tf.all(a, [0, 1]).data(), 0);
  });

  it('2D, axis=0', async () => {
    const a = tf.tensor2d([1, 1, 0, 0], [2, 2], 'bool');
    let r = tf.all(a, 0);

    expect(r.shape).toEqual([2]);
    expectArraysClose(await r.data(), [0, 0]);

    r = tf.all(a, 1);

    expect(r.shape).toEqual([2]);
    expectArraysClose(await r.data(), [1, 0]);
  });

  it('2D, axis=0, keepDims', async () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    const r = a.all(0, true /* keepDims */);

    expect(r.shape).toEqual([1, 3]);
    expectArraysClose(await r.data(), [0, 1, 0]);
  });

  it('2D, axis=1 provided as a number', async () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    const r = tf.all(a, 1);
    expectArraysClose(await r.data(), [0, 0]);
  });

  it('2D, axis = -1 provided as a number', async () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    const r = tf.all(a, -1);
    expectArraysClose(await r.data(), [0, 0]);
  });

  it('2D, axis=[1]', async () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    const r = tf.all(a, [1]);
    expectArraysClose(await r.data(), [0, 0]);
  });

  it('throws when dtype is not boolean', () => {
    const a = tf.tensor2d([1, 1, 0, 0], [2, 2]);
    expect(() => tf.all(a))
        .toThrowError(
            /Argument 'x' passed to 'all' must be bool tensor, but got float/);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.all({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'all' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const a = [0, 0, 0];
    expectArraysClose(await tf.all(a).data(), 0);
  });

  it('throws error for string tensor', () => {
    expect(() => tf.all(['a']))
        .toThrowError(
            /Argument 'x' passed to 'all' must be bool tensor, but got string/);
  });
});

describeWithFlags('Reduction: any', ALL_ENVS, () => {
  it('Tensor1D', async () => {
    let a = tf.tensor1d([0, 0, 0], 'bool');
    expectArraysClose(await tf.any(a).data(), 0);

    a = tf.tensor1d([1, 0, 1], 'bool');
    expectArraysClose(await tf.any(a).data(), 1);

    a = tf.tensor1d([1, 1, 1], 'bool');
    expectArraysClose(await tf.any(a).data(), 1);
  });

  it('ignores NaNs', async () => {
    const a = tf.tensor1d([1, NaN, 0], 'bool');
    expectArraysEqual(await tf.any(a).data(), 1);
  });

  it('2D', async () => {
    const a = tf.tensor2d([1, 1, 0, 0], [2, 2], 'bool');
    expectArraysClose(await tf.any(a).data(), 1);
  });

  it('2D axis=[0,1]', async () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    expectArraysClose(await tf.any(a, [0, 1]).data(), 1);
  });

  it('2D, axis=0', async () => {
    const a = tf.tensor2d([1, 1, 0, 0], [2, 2], 'bool');
    let r = tf.any(a, 0);

    expect(r.shape).toEqual([2]);
    expectArraysClose(await r.data(), [1, 1]);

    r = tf.any(a, 1);

    expect(r.shape).toEqual([2]);
    expectArraysClose(await r.data(), [1, 0]);
  });

  it('2D, axis=0, keepDims', async () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    const r = a.any(0, true /* keepDims */);

    expect(r.shape).toEqual([1, 3]);
    expectArraysClose(await r.data(), [1, 1, 0]);
  });

  it('2D, axis=1 provided as a number', async () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    const r = tf.any(a, 1);
    expectArraysClose(await r.data(), [1, 1]);
  });

  it('2D, axis = -1 provided as a number', async () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    const r = tf.any(a, -1);
    expectArraysClose(await r.data(), [1, 1]);
  });

  it('2D, axis=[1]', async () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    const r = tf.any(a, [1]);
    expectArraysClose(await r.data(), [1, 1]);
  });

  it('throws when dtype is not boolean', () => {
    const a = tf.tensor2d([1, 1, 0, 0], [2, 2]);
    expect(() => tf.any(a))
        .toThrowError(
            /Argument 'x' passed to 'any' must be bool tensor, but got float/);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.any({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'any' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const a = [0, 0, 0];
    expectArraysClose(await tf.any(a).data(), 0);
  });

  it('throws error for string tensor', () => {
    expect(() => tf.any(['a']))
        .toThrowError(/Argument 'x' passed to 'any' must be bool tensor/);
  });
});
