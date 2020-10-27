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
import {expectArraysClose} from '../test_util';

describeWithFlags('max', ALL_ENVS, () => {
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

  it('with a large dimension', async () => {
    const aData = new Float32Array(1000);
    aData[0] = 1;
    const a = tf.tensor1d(aData);
    const r = tf.max(a);
    expectArraysClose(await r.data(), 1);
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

  it('axis permutation does not change input', async () => {
    const input = tf.tensor2d([3, -1, 0, 100, -7, 2], [2, 3]);
    const inputDataBefore = await input.data();

    tf.max(input, [1, 0]);

    const inputDataAfter = await input.data();
    expectArraysClose(inputDataBefore, inputDataAfter);
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

  it('max gradient: 3D, axis=1 keepDims=false', async () => {
    const x = tf.ones([2, 1, 250]);
    const axis = 1;
    const gradients = tf.grad(v => tf.max(v, axis))(x);
    expect(gradients.shape).toEqual(x.shape);
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
