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
import {ALL_ENVS, expectArraysClose, expectArraysEqual, WEBGL_ENVS} from '../test_util';

import * as reduce_util from './reduce_util';

describeWithFlags('Reduction: min', ALL_ENVS, () => {
  it('Tensor1D', () => {
    const a = tf.tensor1d([3, -1, 0, 100, -7, 2]);
    expectArraysClose(tf.min(a), -7);
  });

  it('ignores NaNs', () => {
    const a = tf.tensor1d([3, NaN, 2]);
    expectArraysEqual(tf.min(a), 2);
  });

  it('2D', () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    expectArraysClose(tf.min(a), -7);
  });

  it('2D axis=[0,1]', () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    expectArraysClose(tf.min(a, [0, 1]), -7);
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
    expectArraysClose(tf.min([3, -1, 0, 100, -7, 2]), -7);
  });

  it('min gradient: Scalar', () => {
    const x = tf.scalar(42);
    const dy = tf.scalar(-1);
    const gradients = tf.grad(v => tf.min(v))(x, dy);
    expectArraysClose(gradients, tf.scalar(-1));
  });

  it('min gradient: 1D, ties', () => {
    const x = tf.tensor1d([-1, -3, -7, -7]);
    const dy = tf.scalar(-1);
    const gradients = tf.grad(v => tf.min(v))(x, dy);
    expectArraysClose(gradients, tf.tensor1d([0, 0, -1, -1]));
  });

  it('min gradient: 2D, axes=-1, keepDims=false', () => {
    const x = tf.tensor2d([[-0, -20, -10], [10, 30, 20]]);
    const dy = tf.tensor1d([-1, -1]);
    const axis = -1;
    const gradients = tf.grad(v => tf.min(v, axis))(x, dy);
    expectArraysClose(gradients, tf.tensor2d([[0, -1, 0], [-1, 0, 0]]));
  });

  it('min gradient: ties, 2D, axes=-1, keepDims=false', () => {
    const x = tf.tensor2d([[0, -20, -20], [10, 30, 10]]);
    const dy = tf.tensor1d([-1, -1]);
    const axis = -1;
    const gradients = tf.grad(v => tf.min(v, axis))(x, dy);
    expectArraysClose(gradients, tf.tensor2d([[0, -1, -1], [-1, 0, -1]]));
  });

  it('min gradient: 2D, axes=0, keepDims=false', () => {
    const x = tf.tensor2d([[0, 20, 10], [-10, -30, 20]]);
    const dy = tf.tensor1d([-1, -1, -1]);
    const axis = 0;
    const gradients = tf.grad(v => tf.max(v, axis))(x, dy);
    expectArraysClose(gradients, tf.tensor2d([[-1, -1, 0], [0, 0, -1]]));
  });

  it('min gradient: 2D, axes=-1, keepDims=true', () => {
    const x = tf.tensor2d([[0, -20, -10], [10, 30, 20]]);
    const dy = tf.tensor2d([[-1], [-1]]);
    const axis = -1;
    const keepDims = true;
    const gradients = tf.grad(v => tf.min(v, axis, keepDims))(x, dy);
    expectArraysClose(gradients, tf.tensor2d([[0, -1, 0], [-1, 0, 0]]));
  });

  it('min gradient: 2D, axes=0, keepDims=true', () => {
    const x = tf.tensor2d([[0, -20, -10], [10, 30, -20]]);
    const dy = tf.tensor2d([[-1, -1, -1]]);
    const axis = 0;
    const keepDims = true;
    const gradients = tf.grad(v => tf.min(v, axis, keepDims))(x, dy);
    expectArraysClose(gradients, tf.tensor2d([[-1, -1, 0], [0, 0, -1]]));
  });

  it('min gradient: 3D, axes=[1, 2], keepDims=false', () => {
    const x = tf.tensor3d([[[0, -20], [-10, -15]], [[10, 30], [20, 15]]]);
    const dy = tf.tensor1d([-1, -1]);
    const axis = [1, 2];
    const gradients = tf.grad(v => tf.min(v, axis))(x, dy);
    expectArraysClose(
        gradients, tf.tensor3d([[[0, -1], [0, 0]], [[-1, 0], [0, 0]]]));
  });

  it('min gradient: ties, 3D, axes=[1, 2], keepDims=false', () => {
    const x = tf.tensor3d([[[0, -20], [-20, -20]], [[10, 30], [10, 15]]]);
    const dy = tf.tensor1d([-1, -1]);
    const axis = [1, 2];
    const gradients = tf.grad(v => tf.min(v, axis))(x, dy);
    expectArraysClose(
        gradients, tf.tensor3d([[[0, -1], [-1, -1]], [[-1, 0], [-1, 0]]]));
  });

  it('min gradient: 3D, axes=2, keepDims=false', () => {
    const x = tf.tensor3d([[[0, -20], [-10, -15]], [[10, 30], [20, 15]]]);
    const dy = tf.tensor2d([[-1, -1], [-1, -1]]);
    const axis = 2;
    const gradients = tf.grad(v => tf.min(v, axis))(x, dy);
    expectArraysClose(
        gradients, tf.tensor3d([[[0, -1], [0, -1]], [[-1, 0], [0, -1]]]));
  });

  it('min gradient: 3D, axes=2, keepDims=true', () => {
    const x = tf.tensor3d([[[0, -20], [-10, -15]], [[10, 30], [20, 15]]]);
    const dy = tf.tensor3d([[[-1], [-1]], [[-1], [-1]]]);
    const axis = 2;
    const keepDims = true;
    const gradients = tf.grad(v => tf.min(v, axis, keepDims))(x, dy);
    expectArraysClose(
        gradients, tf.tensor3d([[[0, -1], [0, -1]], [[-1, 0], [0, -1]]]));
  });

  it('min gradient: ties, 4D, axes=[1, 2, 3], keepDims=false', () => {
    const x = tf.tensor4d([
      [[[0, -20], [-20, -20]], [[10, 30], [10, 30]]],
      [[[0, 20], [20, 20]], [[-10, -30], [-10, -30]]]
    ]);
    const dy = tf.tensor1d([-1, -1]);
    const axis = [1, 2, 3];
    const gradients = tf.grad(v => tf.min(v, axis))(x, dy);
    expectArraysClose(gradients, tf.tensor4d([
      [[[0, -1], [-1, -1]], [[0, 0], [0, 0]]],
      [[[0, 0], [0, 0]], [[0, -1], [0, -1]]]
    ]));
  });

  it('min gradient: ties, 4D, axes=[2, 3], keepDims=true', () => {
    const x = tf.tensor4d([
      [[[0, -20], [-20, -20]], [[10, 30], [10, 30]]],
      [[[0, 20], [20, 20]], [[-10, -30], [-10, -30]]]
    ]);
    const dy = tf.tensor4d([[[[-1]], [[-2]]], [[[-3]], [[-4]]]]);
    const axis = [2, 3];
    const keepDims = true;
    const gradients = tf.grad(v => tf.min(v, axis, keepDims))(x, dy);
    expectArraysClose(gradients, tf.tensor4d([
      [[[0, -1], [-1, -1]], [[-2, 0], [-2, 0]]],
      [[[-3, 0], [0, 0]], [[0, -4], [0, -4]]]
    ]));
  });

  it('throws error for string tensor', () => {
    expect(() => tf.min(['a']))
        .toThrowError(/Argument 'x' passed to 'min' must be numeric tensor/);
  });
});

describeWithFlags('Reduction: max', ALL_ENVS, () => {
  it('with one element dominating', () => {
    const a = tf.tensor1d([3, -1, 0, 100, -7, 2]);
    const r = tf.max(a);
    expectArraysClose(r, 100);
  });

  it('with all elements being the same', () => {
    const a = tf.tensor1d([3, 3, 3]);
    const r = tf.max(a);
    expectArraysClose(r, 3);
  });

  it('ignores NaNs', () => {
    expectArraysClose(tf.max(tf.tensor1d([3, NaN, 2])), 3);
  });

  it('2D', () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    expectArraysClose(tf.max(a), 100);
  });

  it('2D axis=[0,1]', () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    expectArraysClose(tf.max(a, [0, 1]), 100);
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
    expectArraysClose(r, 100);
  });

  it('max gradient: Scalar', () => {
    const x = tf.scalar(42);
    const dy = tf.scalar(-1);
    const gradients = tf.grad(v => tf.max(v))(x, dy);
    expectArraysClose(gradients, tf.scalar(-1));
  });

  it('max gradient: 1D, ties', () => {
    const x = tf.tensor1d([1, 3, 7, 7]);
    const dy = tf.scalar(-1);
    const gradients = tf.grad(v => tf.max(v))(x, dy);
    expectArraysClose(gradients, tf.tensor1d([0, 0, -1, -1]));
  });

  it('max gradient: 2D, axes=-1, keepDims=false', () => {
    const x = tf.tensor2d([[0, 20, 10], [-10, -30, -20]]);
    const dy = tf.tensor1d([-1, -1]);
    const axis = -1;
    const gradients = tf.grad(v => tf.max(v, axis))(x, dy);
    expectArraysClose(gradients, tf.tensor2d([[0, -1, 0], [-1, 0, 0]]));
  });

  it('max gradient: ties, 2D, axes=-1, keepDims=false', () => {
    const x = tf.tensor2d([[0, 20, 20], [-10, -30, -10]]);
    const dy = tf.tensor1d([-1, -1]);
    const axis = -1;
    const gradients = tf.grad(v => tf.max(v, axis))(x, dy);
    expectArraysClose(gradients, tf.tensor2d([[0, -1, -1], [-1, 0, -1]]));
  });

  it('max gradient: 2D, axes=0, keepDims=false', () => {
    const x = tf.tensor2d([[0, 20, 10], [-10, -30, 20]]);
    const dy = tf.tensor1d([-1, -1, -1]);
    const axis = 0;
    const gradients = tf.grad(v => tf.max(v, axis))(x, dy);
    expectArraysClose(gradients, tf.tensor2d([[-1, -1, 0], [0, 0, -1]]));
  });

  it('max gradient: 2D, axes=-1, keepDims=true', () => {
    const x = tf.tensor2d([[0, 20, 10], [-10, -30, -20]]);
    const dy = tf.tensor2d([[-1], [-1]]);
    const axis = -1;
    const keepDims = true;
    const gradients = tf.grad(v => tf.max(v, axis, keepDims))(x, dy);
    expectArraysClose(gradients, tf.tensor2d([[0, -1, 0], [-1, 0, 0]]));
  });

  it('max gradient: 2D, axes=0, keepDims=true', () => {
    const x = tf.tensor2d([[0, 20, 10], [-10, -30, 20]]);
    const dy = tf.tensor2d([[-1, -1, -1]]);
    const axis = 0;
    const keepDims = true;
    const gradients = tf.grad(v => tf.max(v, axis, keepDims))(x, dy);
    expectArraysClose(gradients, tf.tensor2d([[-1, -1, 0], [0, 0, -1]]));
  });

  it('max gradient: 3D, axes=[1, 2], keepDims=false', () => {
    const x = tf.tensor3d([[[0, 20], [10, 15]], [[-10, -30], [-20, -15]]]);
    const dy = tf.tensor1d([-1, -1]);
    const axis = [1, 2];
    const gradients = tf.grad(v => tf.max(v, axis))(x, dy);
    expectArraysClose(
        gradients, tf.tensor3d([[[0, -1], [0, 0]], [[-1, 0], [0, 0]]]));
  });

  it('max gradient: ties, 3D, axes=[1, 2], keepDims=false', () => {
    const x = tf.tensor3d([[[0, 20], [20, 20]], [[-10, -30], [-10, -15]]]);
    const dy = tf.tensor1d([-1, -1]);
    const axis = [1, 2];
    const gradients = tf.grad(v => tf.max(v, axis))(x, dy);
    expectArraysClose(
        gradients, tf.tensor3d([[[0, -1], [-1, -1]], [[-1, 0], [-1, 0]]]));
  });

  it('max gradient: 3D, axes=2, keepDims=false', () => {
    const x = tf.tensor3d([[[0, 20], [10, 15]], [[-10, -30], [-20, -15]]]);
    const dy = tf.tensor2d([[-1, -1], [-1, -1]]);
    const axis = 2;
    const gradients = tf.grad(v => tf.max(v, axis))(x, dy);
    expectArraysClose(
        gradients, tf.tensor3d([[[0, -1], [0, -1]], [[-1, 0], [0, -1]]]));
  });

  it('max gradient: 3D, axes=2, keepDims=true', () => {
    const x = tf.tensor3d([[[0, 20], [10, 15]], [[-10, -30], [-20, -15]]]);
    const dy = tf.tensor3d([[[-1], [-1]], [[-1], [-1]]]);
    const axis = 2;
    const keepDims = true;
    const gradients = tf.grad(v => tf.max(v, axis, keepDims))(x, dy);
    expectArraysClose(
        gradients, tf.tensor3d([[[0, -1], [0, -1]], [[-1, 0], [0, -1]]]));
  });

  it('max gradient: ties, 4D, axes=[1, 2, 3], keepDims=false', () => {
    const x = tf.tensor4d([
      [[[0, 20], [20, 20]], [[-10, -30], [-10, -30]]],
      [[[0, -20], [-20, -20]], [[10, 30], [10, 30]]]
    ]);
    const dy = tf.tensor1d([-1, -1]);
    const axis = [1, 2, 3];
    const gradients = tf.grad(v => tf.max(v, axis))(x, dy);
    expectArraysClose(gradients, tf.tensor4d([
      [[[0, -1], [-1, -1]], [[0, 0], [0, 0]]],
      [[[0, 0], [0, 0]], [[0, -1], [0, -1]]]
    ]));
  });

  it('max gradient: ties, 4D, axes=[2, 3], keepDims=true', () => {
    const x = tf.tensor4d([
      [[[0, 20], [20, 20]], [[-10, -30], [-10, -30]]],
      [[[0, -20], [-20, -20]], [[10, 30], [10, 30]]]
    ]);
    const dy = tf.tensor4d([[[[-1]], [[-2]]], [[[-3]], [[-4]]]]);
    const axis = [2, 3];
    const keepDims = true;
    const gradients = tf.grad(v => tf.max(v, axis, keepDims))(x, dy);
    expectArraysClose(gradients, tf.tensor4d([
      [[[0, -1], [-1, -1]], [[-2, 0], [-2, 0]]],
      [[[-3, 0], [0, 0]], [[0, -4], [0, -4]]]
    ]));
  });

  it('throws error for string tensor', () => {
    expect(() => tf.max(['a']))
        .toThrowError(/Argument 'x' passed to 'max' must be numeric tensor/);
  });
});

describeWithFlags('Reduction: argmax', ALL_ENVS, () => {
  it('Tensor1D', () => {
    const a = tf.tensor1d([1, 0, 3, 2]);
    const result = tf.argMax(a);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(result, 2);
  });

  it('one value', () => {
    const a = tf.tensor1d([10]);
    const result = tf.argMax(a);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(result, 0);
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
    expectArraysEqual(result, n - 1);
  });

  it('3D, N > than parallelization threshold', () => {
    const n = reduce_util.PARALLELIZE_THRESHOLD * 2;
    const values = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      values[i] = i;
    }
    const a = tf.tensor3d(values, [1, 1, n]);
    const result = tf.argMax(a, -1);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(result, n - 1);
  });

  it('max index corresponds to start of a non-initial window', () => {
    const n = reduce_util.PARALLELIZE_THRESHOLD * 2;
    const windowSize = reduce_util.computeOptimalWindowSize(n);
    const values = new Float32Array(n);
    const index = windowSize * 2;
    values[index] = 1;
    const a = tf.tensor1d(values);
    const result = tf.argMax(a);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(result, index);
  });

  it('5D, max index corresponds to start of a non-initial window', () => {
    const n = reduce_util.PARALLELIZE_THRESHOLD * 2;
    const windowSize = reduce_util.computeOptimalWindowSize(n);
    const values = new Float32Array(n);
    const index = windowSize * 2;
    values[index] = 1;
    const a = tf.tensor5d(values, [1, 1, 1, 1, n]);
    const result = tf.argMax(a, -1);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(result, index);
  });

  it('ignores NaNs', () => {
    const a = tf.tensor1d([0, 3, 5, NaN, 3]);
    const res = tf.argMax(a);
    expect(res.dtype).toBe('int32');
    expectArraysEqual(res, 2);
  });

  it('2D, no axis specified', () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    expectArraysEqual(tf.argMax(a), [1, 0, 1]);
  });

  it('4D, no axis specified', () => {
    const a = tf.tensor4d([3, -1, 0, 100, -7, 2], [2, 1, 1, 3]);
    expectArraysEqual(tf.argMax(a), [1, 0, 1]);
  });

  it('2D, axis=0', () => {
    const a = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    const r = tf.argMax(a, 0);

    expect(r.shape).toEqual([3]);
    expect(r.dtype).toBe('int32');
    expectArraysEqual(r, [1, 0, 1]);
  });

  it('6D, axis=0', () => {
    const a = tf.tensor6d([3, -1, 0, 100, -7, 2], [2, 1, 1, 1, 1, 3]);
    const r = tf.argMax(a, 0);

    expect(r.shape).toEqual([1, 1, 1, 1, 3]);
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
    expectArraysEqual(result, 2);
  });

  it('accepts tensor with bool values', () => {
    const t = tf.tensor1d([0, 1], 'bool');
    const result = tf.argMax(t);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(result, 1);
  });

  it('has gradient', () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const dy = tf.ones([3], 'float32') as tf.Tensor1D;
    const da = tf.grad((x: tf.Tensor2D) => tf.argMax(x))(a, dy);

    expect(da.dtype).toBe('float32');
    expect(da.shape).toEqual([2, 3]);
    expectArraysClose(da, [0, 0, 0, 0, 0, 0]);
  });

  it('throws error for string tensor', () => {
    expect(() => tf.argMax(['a']))
        .toThrowError(/Argument 'x' passed to 'argMax' must be numeric tensor/);
  });
});

describeWithFlags('Reduction: webgl packed input', WEBGL_ENVS, () => {
  it('argmax 3D, odd number of rows, axis = -1', () => {
    const webglLazilyUnpackFlagSaved = tf.ENV.get('WEBGL_LAZILY_UNPACK');
    tf.ENV.set('WEBGL_LAZILY_UNPACK', true);
    const webglPackBinaryOperationsFlagSaved =
        tf.ENV.get('WEBGL_PACK_BINARY_OPERATIONS');
    tf.ENV.set('WEBGL_PACK_BINARY_OPERATIONS', true);

    const a = tf.tensor3d([3, 2, 5, 100, -7, 2], [2, 1, 3]).add(1);
    const r = tf.argMax(a, -1);
    tf.ENV.set('WEBGL_LAZILY_UNPACK', webglLazilyUnpackFlagSaved);
    tf.ENV.set(
        'WEBGL_PACK_BINARY_OPERATIONS', webglPackBinaryOperationsFlagSaved);

    expect(r.dtype).toBe('int32');
    expectArraysEqual(r, [2, 0]);
  });

  it('argmin 4D, odd number of rows, axis = -1', () => {
    const webglLazilyUnpackFlagSaved = tf.ENV.get('WEBGL_LAZILY_UNPACK');
    tf.ENV.set('WEBGL_LAZILY_UNPACK', true);
    const webglPackBinaryOperationsFlagSaved =
        tf.ENV.get('WEBGL_PACK_BINARY_OPERATIONS');
    tf.ENV.set('WEBGL_PACK_BINARY_OPERATIONS', true);

    const a =
        tf.tensor4d(
              [3, 2, 5, 100, -7, 2, 8, 7, -5, 101, 7, -2, 100, -7, 2, 8, 7, -5],
              [1, 2, 3, 3])
            .add(1);
    const r = tf.argMin(a, -1);
    tf.ENV.set('WEBGL_LAZILY_UNPACK', webglLazilyUnpackFlagSaved);
    tf.ENV.set(
        'WEBGL_PACK_BINARY_OPERATIONS', webglPackBinaryOperationsFlagSaved);

    expect(r.dtype).toBe('int32');
    expectArraysEqual(r, [1, 1, 2, 2, 1, 2]);
  });

  it('should not leak memory when called after unpacked op', () => {
    const webglPackBinaryOperationsFlagSaved =
        tf.ENV.get('WEBGL_PACK_BINARY_OPERATIONS');
    tf.ENV.set('WEBGL_PACK_BINARY_OPERATIONS', false);

    const a =
        tf.tensor5d(
              [3, 2, 5, 100, -7, 2, 8, 7, -5, 101, 7, -2, 100, -7, 2, 8, 7, -5],
              [1, 2, 3, 1, 3])
            .add(1);
    const startNumBytes = tf.memory().numBytes;
    const startNumTensors = tf.memory().numTensors;
    const r = tf.argMin(a, -1);
    tf.ENV.set(
        'WEBGL_PACK_BINARY_OPERATIONS', webglPackBinaryOperationsFlagSaved);
    const endNumBytes = tf.memory().numBytes;
    const endNumTensors = tf.memory().numTensors;
    expect(endNumBytes - startNumBytes).toEqual(24);
    expect(endNumTensors - startNumTensors).toEqual(1);
    expect(r.dtype).toBe('int32');
    expectArraysEqual(r, [1, 1, 2, 2, 1, 2]);
  });
});

describeWithFlags('Reduction: argmin', ALL_ENVS, () => {
  it('Tensor1D', () => {
    const a = tf.tensor1d([1, 0, 3, 2]);
    const result = tf.argMin(a);
    expectArraysEqual(result, 1);
  });

  it('one value', () => {
    const a = tf.tensor1d([10]);
    const result = tf.argMin(a);
    expectArraysEqual(result, 0);
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
    expectArraysEqual(result, n - 1);
  });

  it('4D, N > than parallelization threshold', () => {
    const n = reduce_util.PARALLELIZE_THRESHOLD * 2;
    const values = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      values[i] = n - i;
    }
    const a = tf.tensor4d(values, [1, 1, 1, n]);
    const result = tf.argMin(a, -1);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(result, n - 1);
  });

  it('min index corresponds to start of a non-initial window', () => {
    const n = reduce_util.PARALLELIZE_THRESHOLD * 2;
    const windowSize = reduce_util.computeOptimalWindowSize(n);
    const values = new Float32Array(n);
    const index = windowSize * 2;
    values[index] = -1;
    const a = tf.tensor1d(values);
    const result = tf.argMin(a);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(result, index);
  });

  it('ignores NaNs', () => {
    const a = tf.tensor1d([5, 0, NaN, -1, 3]);
    const res = tf.argMin(a);
    expectArraysEqual(res, 3);
  });

  it('3D, ignores NaNs', () => {
    const a = tf.tensor3d([5, 0, NaN, -1, 3], [1, 1, 5]);
    const res = tf.argMin(a, -1);
    expectArraysEqual(res, 3);
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
    expectArraysEqual(result, 1);
  });

  it('accepts tensor with bool values', () => {
    const t = tf.tensor1d([0, 1], 'bool');
    const result = tf.argMin(t);
    expect(result.dtype).toBe('int32');
    expectArraysEqual(result, 0);
  });

  it('has gradient', () => {
    const a = tf.tensor2d([3, 2, 5, 100, -7, 2], [2, 3]);
    const dy = tf.ones([3], 'float32') as tf.Tensor1D;
    const da = tf.grad((x: tf.Tensor2D) => tf.argMin(x))(a, dy);

    expect(da.dtype).toBe('float32');
    expect(da.shape).toEqual([2, 3]);
    expectArraysClose(da, [0, 0, 0, 0, 0, 0]);
  });

  it('throws error for string tensor', () => {
    expect(() => tf.argMin(['a']))
        .toThrowError(/Argument 'x' passed to 'argMin' must be numeric tensor/);
  });
});

describeWithFlags('Reduction: logSumExp', ALL_ENVS, () => {
  it('0', () => {
    const a = tf.scalar(0);
    const result = tf.logSumExp(a);
    expectArraysClose(result, 0);
  });

  it('basic', () => {
    const a = tf.tensor1d([1, 2, -3]);
    const result = tf.logSumExp(a);

    expectArraysClose(
        result, Math.log(Math.exp(1) + Math.exp(2) + Math.exp(-3)));
  });

  it('propagates NaNs', () => {
    const a = tf.tensor1d([1, 2, NaN]);
    const result = tf.logSumExp(a);
    expectArraysEqual(result, NaN);
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
    expectArraysClose(
        result, Math.log(Math.exp(1) + Math.exp(2) + Math.exp(-3)));
  });

  it('throws error for string tensor', () => {
    expect(() => tf.logSumExp(['a']))
        .toThrowError(
            /Argument 'x' passed to 'logSumExp' must be numeric tensor/);
  });
});

describeWithFlags('Reduction: sum', ALL_ENVS, () => {
  it('basic', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const result = tf.sum(a);
    expectArraysClose(result, 7);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor2d([1, 2, 3, NaN, 0, 1], [3, 2]);
    expectArraysEqual(tf.sum(a), NaN);
  });

  it('sum over dtype int32', () => {
    const a = tf.tensor1d([1, 5, 7, 3], 'int32');
    const sum = tf.sum(a);
    expectArraysEqual(sum, 16);
  });

  it('sum over dtype bool', () => {
    const a = tf.tensor1d([true, false, false, true, true], 'bool');
    const sum = tf.sum(a);
    expectArraysEqual(sum, 3);
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
    expectArraysClose(result, 7);
  });

  it('throws error for string tensor', () => {
    expect(() => tf.sum(['a']))
        .toThrowError(/Argument 'x' passed to 'sum' must be numeric tensor/);
  });
});

describeWithFlags('Reduction: prod', ALL_ENVS, () => {
  it('basic', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const result = tf.prod(a);
    expectArraysClose(result, 0);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor2d([1, 2, 3, NaN, 0, 1], [3, 2]);
    expectArraysEqual(tf.prod(a), NaN);
  });

  it('prod over dtype int32', () => {
    const a = tf.tensor1d([1, 5, 7, 3], 'int32');
    const prod = tf.prod(a);
    expectArraysEqual(prod, 105);
  });

  it('prod over dtype bool', () => {
    const a = tf.tensor1d([true, false, false, true, true], 'bool');
    const prod = tf.prod(a);
    expectArraysEqual(prod, 0);
  });

  it('prods all values in 2D array with keep dim', () => {
    const a = tf.tensor2d([1, 2, 3, 1, 0, 1], [3, 2]);
    const res = tf.prod(a, null, true /* keepDims */);

    expect(res.shape).toEqual([1, 1]);
    expectArraysClose(res, 0);
  });

  it('prods across axis=0 in 2D array', () => {
    const a = tf.tensor2d([1, 2, 3, 1, 0, 1], [3, 2]);
    const res = tf.prod(a, [0]);

    expect(res.shape).toEqual([2]);
    expectArraysClose(res, [0, 2]);
  });

  it('prods across axis=0 in 2D array, keepDims', () => {
    const a = tf.tensor2d([1, 2, 3, 1, 0, 1], [3, 2]);
    const res = tf.prod(a, [0], true /* keepDims */);

    expect(res.shape).toEqual([1, 2]);
    expectArraysClose(res, [0, 2]);
  });

  it('prods across axis=1 in 2D array', () => {
    const a = tf.tensor2d([1, 2, 3, 1, 1, 1], [3, 2]);
    const res = tf.prod(a, [1]);

    expect(res.shape).toEqual([3]);
    expectArraysClose(res, [2, 3, 1]);
  });

  it('2D, axis=1 provided as number', () => {
    const a = tf.tensor2d([1, 2, 3, 1, 1, 1], [2, 3]);
    const res = tf.prod(a, 1);

    expect(res.shape).toEqual([2]);
    expectArraysClose(res, [6, 1]);
  });

  it('2D, axis = -1 provided as number', () => {
    const a = tf.tensor2d([1, 2, 3, 1, 1, 1], [2, 3]);
    const res = tf.prod(a, -1);

    expect(res.shape).toEqual([2]);
    expectArraysClose(res, [6, 1]);
  });

  it('prods across axis=0,1 in 2D array', () => {
    const a = tf.tensor2d([1, 2, 3, 1, 1, 1], [3, 2]);
    const res = tf.prod(a, [0, 1]);

    expect(res.shape).toEqual([]);
    expectArraysClose(res, [6]);
  });

  it('2D, axis=[-1,-2] in 2D array', () => {
    const a = tf.tensor2d([1, 2, 3, 1, 1, 1], [3, 2]);
    const res = tf.prod(a, [-1, -2]);

    expect(res.shape).toEqual([]);
    expectArraysClose(res, [6]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.prod({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'prod' must be a Tensor/);
  });

  it('accepts a tensor-like object', () => {
    const result = tf.prod([[1, 2], [3, 1], [1, 1]]);
    expectArraysClose(result, 6);
  });

  it('throws error for string tensor', () => {
    expect(() => tf.prod(['a']))
        .toThrowError(/Argument 'x' passed to 'prod' must be numeric tensor/);
  });
});

describeWithFlags('Reduction: mean', ALL_ENVS, () => {
  it('basic', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const r = tf.mean(a);

    expect(r.dtype).toBe('float32');
    expectArraysClose(r, 7 / 6);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor2d([1, 2, 3, NaN, 0, 1], [3, 2]);
    const r = tf.mean(a);

    expect(r.dtype).toBe('float32');
    expectArraysEqual(r, NaN);
  });

  it('mean(int32) => float32', () => {
    const a = tf.tensor1d([1, 5, 7, 3], 'int32');
    const r = tf.mean(a);

    expect(r.dtype).toBe('float32');
    expectArraysClose(r, 4);
  });

  it('mean(bool) => float32', () => {
    const a = tf.tensor1d([true, false, false, true, true], 'bool');
    const r = tf.mean(a);

    expect(r.dtype).toBe('float32');
    expectArraysClose(r, 3 / 5);
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
    const dyVal = dy.arraySync();
    expect(da.shape).toEqual(a.shape);
    expectArraysClose(da, [
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

  it('accepts a tensor-like object', () => {
    const r = tf.mean([[1, 2, 3], [0, 0, 1]]);

    expect(r.dtype).toBe('float32');
    expectArraysClose(r, 7 / 6);
  });

  it('throws error for string tensor', () => {
    expect(() => tf.mean(['a']))
        .toThrowError(/Argument 'x' passed to 'mean' must be numeric tensor/);
  });
});

describeWithFlags('Reduction: moments', ALL_ENVS, () => {
  it('basic', () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const {mean, variance} = tf.moments(a);

    expect(mean.dtype).toBe('float32');
    expect(variance.dtype).toBe('float32');
    expectArraysClose(mean, 7 / 6);
    expectArraysClose(variance, 1.1389);
  });

  it('propagates NaNs', () => {
    const a = tf.tensor2d([1, 2, 3, NaN, 0, 1], [3, 2]);
    const {mean, variance} = tf.moments(a);

    expect(mean.dtype).toBe('float32');
    expect(variance.dtype).toBe('float32');
    expectArraysEqual(mean, NaN);
    expectArraysEqual(variance, NaN);
  });

  it('moments(int32) => float32', () => {
    const a = tf.tensor1d([1, 5, 7, 3], 'int32');
    const {mean, variance} = tf.moments(a);

    expect(mean.dtype).toBe('float32');
    expect(variance.dtype).toBe('float32');
    expectArraysClose(mean, 4);
    expectArraysClose(variance, 5);
  });

  it('moments(bool) => float32', () => {
    const a = tf.tensor1d([true, false, false, true, true], 'bool');
    const {mean, variance} = tf.moments(a);

    expect(mean.dtype).toBe('float32');
    expect(variance.dtype).toBe('float32');
    expectArraysClose(mean, 3 / 5);
    expectArraysClose(variance, 0.23999998);
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
    expectArraysClose(mean, 7 / 6);
    expectArraysClose(variance, 1.1389);
  });
});

describeWithFlags('Reduction: norm', ALL_ENVS, () => {
  it('scalar norm', () => {
    const a = tf.scalar(-22.0);
    const norm = tf.norm(a);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(norm, 22);
  });

  it('vector inf norm', () => {
    const a = tf.tensor1d([1, -2, 3, -4]);
    const norm = tf.norm(a, Infinity);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(norm, 4);
  });

  it('vector -inf norm', () => {
    const a = tf.tensor1d([1, -2, 3, -4]);
    const norm = tf.norm(a, -Infinity);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(norm, 1);
  });

  it('vector 1 norm', () => {
    const a = tf.tensor1d([1, -2, 3, -4]);
    const norm = tf.norm(a, 1);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(norm, 10);
  });

  it('vector euclidean norm', () => {
    const a = tf.tensor1d([1, -2, 3, -4]);
    const norm = tf.norm(a, 'euclidean');

    expect(norm.dtype).toBe('float32');
    expectArraysClose(norm, 5.4772);
  });

  it('vector 2-norm', () => {
    const a = tf.tensor1d([1, -2, 3, -4]);
    const norm = tf.norm(a, 2);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(norm, 5.4772);
  });

  it('vector >2-norm to throw error', () => {
    const a = tf.tensor1d([1, -2, 3, -4]);
    expect(() => tf.norm(a, 3)).toThrowError();
  });

  it('matrix inf norm', () => {
    const a = tf.tensor2d([1, 2, -3, 1, 0, 1], [3, 2]);
    const norm = tf.norm(a, Infinity, [0, 1]);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(norm, 4);
  });

  it('matrix -inf norm', () => {
    const a = tf.tensor2d([1, 2, -3, 1, 0, 1], [3, 2]);
    const norm = tf.norm(a, -Infinity, [0, 1]);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(norm, 1);
  });

  it('matrix 1 norm', () => {
    const a = tf.tensor2d([1, 2, -3, 1, 1, 1], [3, 2]);
    const norm = tf.norm(a, 1, [0, 1]);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(norm, 5);
  });

  it('matrix euclidean norm', () => {
    const a = tf.tensor2d([1, 2, -3, 1, 1, 1], [3, 2]);
    const norm = tf.norm(a, 'euclidean', [0, 1]);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(norm, 4.123);
  });

  it('matrix fro norm', () => {
    const a = tf.tensor2d([1, 2, -3, 1, 1, 1], [3, 2]);
    const norm = tf.norm(a, 'fro', [0, 1]);

    expect(norm.dtype).toBe('float32');
    expectArraysClose(norm, 4.123);
  });

  it('matrix other norm to throw error', () => {
    const a = tf.tensor2d([1, 2, -3, 1, 1, 1], [3, 2]);
    expect(() => tf.norm(a, 2, [0, 1])).toThrowError();
  });

  it('propagates NaNs for norm', () => {
    const a = tf.tensor2d([1, 2, 3, NaN, 0, 1], [3, 2]);
    const norm = tf.norm(a);

    expect(norm.dtype).toBe('float32');
    expectArraysEqual(norm, NaN);
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
    expectArraysClose(norm, 10);
  });

  it('throws error for string tensors', () => {
    expect(() => tf.norm([
      'a', 'b'
    ])).toThrowError(/Argument 'x' passed to 'norm' must be numeric tensor/);
  });
});

describeWithFlags('Reduction: all', ALL_ENVS, () => {
  it('Tensor1D', () => {
    let a = tf.tensor1d([0, 0, 0], 'bool');
    expectArraysClose(tf.all(a), 0);

    a = tf.tensor1d([1, 0, 1], 'bool');
    expectArraysClose(tf.all(a), 0);

    a = tf.tensor1d([1, 1, 1], 'bool');
    expectArraysClose(tf.all(a), 1);
  });

  it('ignores NaNs', () => {
    const a = tf.tensor1d([1, NaN, 1], 'bool');
    expectArraysEqual(tf.all(a), 1);
  });

  it('2D', () => {
    const a = tf.tensor2d([1, 1, 0, 0], [2, 2], 'bool');
    expectArraysClose(tf.all(a), 0);
  });

  it('2D axis=[0,1]', () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    expectArraysClose(tf.all(a, [0, 1]), 0);
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
    expect(() => tf.all(a))
        .toThrowError(
            /Argument 'x' passed to 'all' must be bool tensor, but got float/);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.all({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'all' must be a Tensor/);
  });

  it('accepts a tensor-like object', () => {
    const a = [0, 0, 0];
    expectArraysClose(tf.all(a), 0);
  });

  it('throws error for string tensor', () => {
    expect(() => tf.all(['a']))
        .toThrowError(
            /Argument 'x' passed to 'all' must be bool tensor, but got string/);
  });
});

describeWithFlags('Reduction: any', ALL_ENVS, () => {
  it('Tensor1D', () => {
    let a = tf.tensor1d([0, 0, 0], 'bool');
    expectArraysClose(tf.any(a), 0);

    a = tf.tensor1d([1, 0, 1], 'bool');
    expectArraysClose(tf.any(a), 1);

    a = tf.tensor1d([1, 1, 1], 'bool');
    expectArraysClose(tf.any(a), 1);
  });

  it('ignores NaNs', () => {
    const a = tf.tensor1d([1, NaN, 0], 'bool');
    expectArraysEqual(tf.any(a), 1);
  });

  it('2D', () => {
    const a = tf.tensor2d([1, 1, 0, 0], [2, 2], 'bool');
    expectArraysClose(tf.any(a), 1);
  });

  it('2D axis=[0,1]', () => {
    const a = tf.tensor2d([1, 1, 0, 0, 1, 0], [2, 3], 'bool');
    expectArraysClose(tf.any(a, [0, 1]), 1);
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
    expect(() => tf.any(a))
        .toThrowError(
            /Argument 'x' passed to 'any' must be bool tensor, but got float/);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.any({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'any' must be a Tensor/);
  });

  it('accepts a tensor-like object', () => {
    const a = [0, 0, 0];
    expectArraysClose(tf.any(a), 0);
  });

  it('throws error for string tensor', () => {
    expect(() => tf.any(['a']))
        .toThrowError(/Argument 'x' passed to 'any' must be bool tensor/);
  });
});
