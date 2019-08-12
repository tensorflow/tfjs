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
import {expectArraysClose} from '../test_util';

describeWithFlags('maxPool', ALL_ENVS, () => {
  it('x=[1,1,1] f=[1,1] s=1 [0] => [0]', async () => {
    const x = tf.tensor3d([0], [1, 1, 1]);

    const result = tf.maxPool(x, 1, 1, 0);

    expectArraysClose(await result.data(), [0]);
  });

  it('x=[3,3,1] f=[2,2] s=1, p=0', async () => {
    // Feed forward.
    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 9, 8], [3, 3, 1]);

    const result = tf.maxPool(x, 2, 1, 0);

    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(await result.data(), [5, 6, 9, 9]);
  });

  it('x=[3,3,1] f=[2,2] s=1 p=same', async () => {
    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 9, 8], [3, 3, 1]);

    const result = tf.maxPool(x, 2, 1, 'same');
    const resultData = await result.data();

    tf.test_util.expectArraysClose(
        resultData, new Float32Array([5, 6, 6, 9, 9, 8, 9, 9, 8]));
  });

  it('x=[2,3,3,1] f=[2,2] s=1', async () => {
    // Feed forward.
    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 9, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 3, 1]);

    const result = tf.maxPool(x, 2, 1, 0);

    expect(result.shape).toEqual([2, 2, 2, 1]);
    expectArraysClose(await result.data(), [5, 6, 9, 9, 5, 6, 8, 9]);
  });

  it('[x=[3,3,1] f=[2,2] s=1 ignores NaNs', async () => {
    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, NaN, 9], [3, 3, 1]);

    const result = tf.maxPool(x, 2, 1, 0);

    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(await result.data(), [5, 6, 7, 9]);
  });

  it('x=[3,3,2] f=[2,2] s=1', async () => {
    // Feed forward.
    const x = tf.tensor3d(
        [1, 99, 2, 88, 3, 77, 4, 66, 5, 55, 6, 44, 7, 33, 9, 22, 8, 11],
        [3, 3, 2]);

    const result = tf.maxPool(x, 2, 1, 0);

    expect(result.shape).toEqual([2, 2, 2]);
    expectArraysClose(await result.data(), [5, 99, 6, 88, 9, 66, 9, 55]);
  });

  it('x=[4,4,1] f=[2,2] s=2', async () => {
    // Feed forward.
    const x = tf.tensor3d(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [4, 4, 1]);

    const result = tf.maxPool(x, 2, 2, 0);

    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(await result.data(), [5, 7, 13, 15]);
  });

  it('x=[2,2,1] f=[2,2] s=1 p=same', async () => {
    // Feed forward.
    const x = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const fSize = 2;
    const strides = 1;
    const result = tf.maxPool(x, fSize, strides, 'same');
    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(await result.data(), [4, 4, 4, 4]);
  });

  it('x=[2,2,3] f=[1,1] s=2 p=1 dimRoundingMode=floor', () => {
    // Feed forward.
    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 2, 3]);
    const result = tf.maxPool(x, 1, 2, 1, 'floor');

    expect(result.shape).toEqual([2, 2, 3]);
  });

  it('throws when x is not rank 3', () => {
    // tslint:disable-next-line:no-any
    const x: any = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]);

    expect(() => tf.maxPool(x, 2, 1, 0)).toThrowError();
  });

  it('throws when dimRoundingMode is set and pad is not a number', () => {
    const x = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);

    const pad = 'valid';
    const dimRoundingMode = 'round';

    expect(() => tf.maxPool(x, 2, 1, pad, dimRoundingMode)).toThrowError();
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.maxPool({} as tf.Tensor3D, 2, 1, 'valid'))
        .toThrowError(/Argument 'x' passed to 'maxPool' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const x = [[[0]]];  // 1x1x1
    const result = tf.maxPool(x, 1, 1, 0);
    expectArraysClose(await result.data(), [0]);
  });
});

describeWithFlags('maxPoolBackprop', ALL_ENVS, () => {
  it('gradients x=[3,3,1] f=[2,2] s=1 no dup max value, test #1', async () => {
    const dy = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3, 1]);
    const expected = [0, 0, 0, 0, 1, 2, 0, 3, 4];

    const dx = tf.grad((x: tf.Tensor3D) => x.maxPool(2, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient with clones x=[3,3,1] f=[2,2] s=1', async () => {
    const dy = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3, 1]);
    const expected = [0, 0, 0, 0, 1, 2, 0, 3, 4];

    const dx = tf.grad(
        (x: tf.Tensor3D) => tf.maxPool(x.clone(), 2, 1, 0).clone())(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradients x=[3,3,1] f=[2,2] s=1 no dup max value, test #2', async () => {
    const dy = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const x = tf.tensor3d([9, 5, 6, 6, 8, 4, 9, 5, 10], [3, 3, 1]);
    const expected = [1, 0, 0, 0, 2, 0, 3, 0, 4];

    const dx = tf.grad((x: tf.Tensor3D) => x.maxPool(2, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradients x=[2,3,3,1] f=[2,2] s=1 no duplicate max value', async () => {
    // This test batches the [3,3,1] tests.
    const dy = tf.tensor4d([1, 2, 3, 4, 1, 2, 3, 4], [2, 2, 2, 1]);
    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 5, 6, 6, 8, 4, 9, 5, 10], [2, 3, 3, 1]);
    const expected = [0, 0, 0, 0, 1, 2, 0, 3, 4, 1, 0, 0, 0, 2, 0, 3, 0, 4];

    const dx = tf.grad((x: tf.Tensor4D) => x.maxPool(2, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[3,3,1] f=[2,2] s=1 dup max value, test 1', async () => {
    const dy = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const x = tf.tensor3d([0, 0, 0, 0, 5, 0, 0, 0, 0], [3, 3, 1]);
    const expected = [0, 0, 0, 0, 10, 0, 0, 0, 0];

    const dx = tf.grad((x: tf.Tensor3D) => x.maxPool(2, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[3,3,1] f=[2,2] s=1 dup max value, test 2', async () => {
    const dy = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const x = tf.tensor3d([1, 3, 2, 1, 2, 1, 1, 1, 5], [3, 3, 1]);
    const expected = [0, 3, 0, 0, 3, 0, 0, 0, 4];

    const dx = tf.grad((x: tf.Tensor3D) => x.maxPool(2, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[2,3,3,1] f=[2,2] s=1 dup max value in 2nd input',
     async () => {
       const dy = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);
       const x = tf.tensor4d(
           [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 9, 8],
           [2, 3, 3, 1]);
       const expected = new Float32Array(
           [0, 0, 0, 0, 1, 2, 0, 3, 4, 0, 0, 0, 0, 5, 6, 0, 15, 0]);

       const dx = tf.grad((x: tf.Tensor4D) => x.maxPool(2, 1, 0))(x, dy);

       expect(dx.shape).toEqual(x.shape);
       expectArraysClose(await dx.data(), expected);
     });

  it('gradient x=[4,4,1] f=[2,2] s=2 test #1', async () => {
    const dy = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const x = tf.tensor3d(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [4, 4, 1]);
    const expected = [0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 3, 0, 4];

    const dx = tf.grad((x: tf.Tensor3D) => x.maxPool(2, 2, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[4,4,1] f=[2,2] s=2 test #2', async () => {
    const dy = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const x = tf.tensor3d(
        [1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1], [4, 4, 1]);
    const expected = [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0];

    const dx = tf.grad((x: tf.Tensor3D) => x.maxPool(2, 2, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[5,5,1] f=[3,3] s=2 no duplicate max value', async () => {
    const dy = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const x = tf.tensor3d(
        [
          0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        ],
        [5, 5, 1]);
    const expected = [
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 4
    ];

    const dx = tf.grad((x: tf.Tensor3D) => x.maxPool(3, 2, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[5,5,1] f=[3,3] s=2 duplicate max value', async () => {
    const dy = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const x = tf.tensor3d(
        [
          0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 24,
          13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 12
        ],
        [5, 5, 1]);
    const expected = [
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ];

    const dx = tf.grad((x: tf.Tensor3D) => x.maxPool(3, 2, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  // Max pool backprop depth > 1.
  it('gradient x=[3,3,2] f=[2,2] s=1, no duplicate max value', async () => {
    // This test combines the first two 3x3x1 tests with no duplicates to
    // make depth=2,
    // dy is slightly modified to show the difference.
    const dy = tf.tensor3d([1, 44, 2, 33, 3, 22, 4, 11], [2, 2, 2]);
    const x = tf.tensor3d(
        [1, 99, 2, 55, 3, 66, 4, 66, 5, 88, 6, 44, 7, 99, 8, 55, 9, 100],
        [3, 3, 2]);
    const expected = [0, 44, 0, 0, 0, 0, 0, 0, 1, 33, 2, 0, 0, 22, 3, 0, 4, 11];

    const dx = tf.grad((x: tf.Tensor3D) => x.maxPool(2, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[3,3,2] f=[2,2] s=1 duplicate max value', async () => {
    // This test combines the first two 3x3x1 tests with duplicates to
    // make depth=2,
    // dy is slightly modified to show the difference.
    const dy = tf.tensor3d([1, 44, 2, 33, 3, 22, 4, 11], [2, 2, 2]);
    const x = tf.tensor3d(
        [0, 1, 0, 3, 0, 2, 0, 1, 5, 2, 0, 1, 0, 1, 0, 1, 0, 5], [3, 3, 2]);
    const expected = new Float32Array(
        [0, 0, 0, 77, 0, 0, 0, 0, 10, 22, 0, 0, 0, 0, 0, 0, 0, 11]);

    const dx = tf.grad((x: tf.Tensor3D) => x.maxPool(2, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[4,4,2] f=[2,2] s=1', async () => {
    // This test combines the first two 4x4x1 tests with duplicates to make
    // depth=2,
    // dy is slightly modified to show the difference.
    const dy = tf.tensor3d([1, 11, 2, 22, 3, 33, 4, 44], [2, 2, 2]);
    const x = tf.tensor3d(
        [
          0, 1, 1, 2, 2,  2, 3,  1, 4,  1, 5,  1, 6,  1, 7,  1,
          8, 1, 9, 1, 10, 1, 11, 1, 12, 1, 13, 2, 14, 2, 15, 1
        ],
        [4, 4, 2]);
    const expected = [
      0, 0, 0, 11, 0, 22, 0, 0, 0, 0, 1, 0,  0, 0,  2, 0,
      0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 3, 33, 0, 44, 4, 0
    ];

    const dx = tf.grad((x: tf.Tensor3D) => x.maxPool(2, 2, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=5x5x2, f=3, s=2 no duplicate max value', async () => {
    // This test combines the first two 5x5x1 tests with duplicates to make
    // depth=2,
    // dy is slightly modified to show the difference.
    const dy = tf.tensor3d([1, 11, 2, 22, 3, 33, 4, 44], [2, 2, 2]);
    const x = tf.tensor3d(
        [
          0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,
          8,  9,  9,  10, 10, 11, 11, 12, 24, 13, 13, 14, 14, 15, 15, 16, 16,
          17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 12
        ],
        [5, 5, 2]);
    const expected = [
      0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 110, 0, 0, 2, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 3, 0, 0, 0, 4, 0
    ];

    const dx = tf.grad((x: tf.Tensor3D) => x.maxPool(3, 2, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });
});

describeWithFlags('avgPool', ALL_ENVS, () => {
  it('x=[1,1,1] f=[1,1] s=1 [0] => [0]', async () => {
    const a = tf.tensor3d([0], [1, 1, 1]);
    const result = tf.avgPool(a, 1, 1, 0);
    expectArraysClose(await result.data(), [0]);
  });

  it('x=[3,3,1] f=[2,2] s=1', async () => {
    // Feed forward.
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 9, 8], [3, 3, 1]);
    const result = tf.avgPool(a, 2, 1, 0);

    expect(result.shape).toEqual([2, 2, 1]);
    expect(result.dtype).toBe('float32');
    expectArraysClose(await result.data(), [3, 4, 6.25, 7]);
  });

  it('input int32 throws error', () => {
    // Feed forward.
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 9, 8], [3, 3, 1], 'int32');
    expect(() => tf.avgPool(a, 2, 1, 0)).toThrowError();
  });

  it('x=[2,3,3,1] f=[2,2], s=1', async () => {
    // Feed forward.
    const a = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 9, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 3, 1]);
    const result = tf.avgPool(a, 2, 1, 0);

    expect(result.shape).toEqual([2, 2, 2, 1]);
    expectArraysClose(await result.data(), [3, 4, 6.25, 7, 3, 4, 6, 7]);
  });

  it('x=[3,3,1] f=[2,2] s=1 propagates NaNs', async () => {
    // Feed forward.
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, NaN, 8], [3, 3, 1]);
    const result = tf.avgPool(a, 2, 1, 0);

    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(await result.data(), [3, 4, NaN, NaN]);
  });

  it('x=[3,3,2] f=[2,2] s=1', async () => {
    // Feed forward.
    const a = tf.tensor3d(
        [1, 99, 2, 88, 3, 77, 4, 66, 5, 55, 6, 44, 7, 33, 9, 22, 8, 11],
        [3, 3, 2]);
    const result = tf.avgPool(a, 2, 1, 0);

    expect(result.shape).toEqual([2, 2, 2]);
    expectArraysClose(await result.data(), [3, 77, 4, 66, 6.25, 44, 7, 33]);
  });

  it('x=[4,4,1] f=[2,2] s=2', async () => {
    // Feed forward.
    const a = tf.tensor3d(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [4, 4, 1]);
    const result = tf.avgPool(a, 2, 2, 0);

    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(await result.data(), [2.5, 4.5, 10.5, 12.5]);
  });

  it('x=[2,2,1] f=[2,2] s=1 p=same', async () => {
    // Feed forward.
    const a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const fSize = 2;
    const strides = 1;
    const result = tf.avgPool(a, fSize, strides, 'same');

    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(await result.data(), [2.5, 3, 3.5, 4]);
  });

  it('x=[2,2,3] f=[1,1] s=2 p=1 dimRoundingMode=floor', () => {
    // Feed forward.
    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 2, 3]);
    const result = tf.avgPool(x, 1, 2, 1, 'floor');

    expect(result.shape).toEqual([2, 2, 3]);
  });

  it('gradient x=[1,1,1] f=[1,1] s=1 [0] => [0]', async () => {
    const x = tf.tensor3d([0], [1, 1, 1]);
    const dy = tf.tensor3d([0], [1, 1, 1]);
    const dx = tf.grad((x: tf.Tensor3D) => x.avgPool(1, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), [0]);
  });

  it('gradient with clones', async () => {
    const x = tf.tensor3d([0], [1, 1, 1]);
    const dy = tf.tensor3d([0], [1, 1, 1]);
    const dx = tf.grad(
        (x: tf.Tensor3D) => tf.avgPool(x.clone(), 1, 1, 0).clone())(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), [0]);
  });

  it('gradient x=[3,3,1] f=[2,2] s=1', async () => {
    // Feed forward.
    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 9, 8], [3, 3, 1]);
    const dy = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const avgMultiplier = 1 / (2 * 2);

    const dx = tf.grad((x: tf.Tensor3D) => x.avgPool(2, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), [
      1 * avgMultiplier, 3 * avgMultiplier, 2 * avgMultiplier,
      4 * avgMultiplier, 10 * avgMultiplier, 6 * avgMultiplier,
      3 * avgMultiplier, 7 * avgMultiplier, 4 * avgMultiplier
    ]);
  });

  it('gradient x=[2,3,3,1] f=[2,2], s=1', async () => {
    // Feed forward.
    const x = tf.tensor4d(
        [1, 2, 3, 4, 5, 6, 7, 9, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 3, 1]);
    const dy = tf.tensor4d([1, 2, 3, 4, 1, 2, 3, 4], [2, 2, 2, 1]);
    const avgMultiplier = 1 / (2 * 2);

    const dx = tf.grad((x: tf.Tensor4D) => x.avgPool(2, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), [
      1 * avgMultiplier, 3 * avgMultiplier, 2 * avgMultiplier,
      4 * avgMultiplier, 10 * avgMultiplier, 6 * avgMultiplier,
      3 * avgMultiplier, 7 * avgMultiplier, 4 * avgMultiplier,
      1 * avgMultiplier, 3 * avgMultiplier, 2 * avgMultiplier,
      4 * avgMultiplier, 10 * avgMultiplier, 6 * avgMultiplier,
      3 * avgMultiplier, 7 * avgMultiplier, 4 * avgMultiplier
    ]);
  });

  it('throws when dimRoundingMode is set and pad is not a number', () => {
    const x = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);

    const pad = 'valid';
    const dimRoundingMode = 'round';

    expect(() => tf.avgPool(x, 2, 1, pad, dimRoundingMode)).toThrowError();
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.avgPool({} as tf.Tensor3D, 2, 1, 'valid'))
        .toThrowError(/Argument 'x' passed to 'avgPool' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const a = [[[0]]];  // 1x1x1
    const result = tf.avgPool(a, 1, 1, 0);
    expectArraysClose(await result.data(), [0]);
  });
});

describeWithFlags('pool', ALL_ENVS, () => {
  // First test that tf.pool calls are consistent with maxPool/avgPool by
  // duplicating some maxPool/avgPool tests. The implementation code is the
  // same, so we don't need the same level of thoroughness here.

  it('max x=[1,1,1] f=[1,1] s=1 d=1 [0] => [0]', async () => {
    const x = tf.tensor3d([0], [1, 1, 1]);

    const windowShape = 1;
    const padding = 0;

    const result = tf.pool(x, windowShape, 'max', padding);
    expectArraysClose(await result.data(), [0]);
  });

  it('max x=[3,3,1] f=[2,2] s=1 d=1', async () => {
    // Feed forward.
    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 9, 8], [3, 3, 1]);

    const windowShape = 2;
    const padding = 0;
    const dilationRate: number = undefined;
    const strides: number = undefined;

    const result =
        tf.pool(x, windowShape, 'max', padding, dilationRate, strides);

    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(await result.data(), [5, 6, 9, 9]);
  });

  it('max x=[4,4,1] f=[2,2] s=2 d=1', async () => {
    // Feed forward.
    const x = tf.tensor3d(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [4, 4, 1]);

    const windowShape = 2;
    const padding = 0;
    const dilationRate: number = undefined;
    const strides = 2;
    const result =
        tf.pool(x, windowShape, 'max', padding, dilationRate, strides);

    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(await result.data(), [5, 7, 13, 15]);
  });

  it('max x=[2,2,1] f=[2,2] s=1 d=1 p=same', async () => {
    // Feed forward.
    const x = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);

    const windowShape = 2;
    const padding = 'same';
    const dilationRate: number = undefined;
    const strides = 1;

    const result =
        tf.pool(x, windowShape, 'max', padding, dilationRate, strides);
    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(await result.data(), [4, 4, 4, 4]);
  });

  it('avg x=[1,1,1] f=[1,1] s=1 d=1 [0] => [0]', async () => {
    const a = tf.tensor3d([0], [1, 1, 1]);

    const windowShape = 1;
    const padding = 0;

    const result = tf.pool(a, windowShape, 'avg', padding);
    expectArraysClose(await result.data(), [0]);
  });

  it('avg x=[3,3,1] f=[2,2] s=1 d=1', async () => {
    // Feed forward.
    const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 9, 8], [3, 3, 1]);

    const windowShape = 2;
    const padding = 0;
    const dilationRate: number = undefined;
    const strides: number = undefined;

    const result =
        tf.pool(a, windowShape, 'avg', padding, dilationRate, strides);

    expect(result.shape).toEqual([2, 2, 1]);
    expect(result.dtype).toBe('float32');
    expectArraysClose(await result.data(), [3, 4, 6.25, 7]);
  });

  it('avg x=[4,4,1] f=[2,2] s=2 d=1', async () => {
    // Feed forward.
    const a = tf.tensor3d(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [4, 4, 1]);

    const windowShape = 2;
    const padding = 0;
    const dilationRate: number = undefined;
    const strides = 2;

    const result =
        tf.pool(a, windowShape, 'avg', padding, dilationRate, strides);

    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(await result.data(), [2.5, 4.5, 10.5, 12.5]);
  });

  it('avg x=[2,2,1] f=[2,2] s=1 p=same', async () => {
    // Feed forward.
    const a = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);

    const windowShape = 2;
    const padding = 'same';
    const dilationRate: number = undefined;
    const strides = 1;

    const result =
        tf.pool(a, windowShape, 'avg', padding, dilationRate, strides);

    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(await result.data(), [2.5, 3, 3.5, 4]);
  });

  // tf.pool supports dilation, unlike maxPool or avgPool
  it('max x=[4,3,1] f=[2,2] s=1 d=2', async () => {
    // Feed forward.
    const x = tf.tensor3d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [4, 4, 1]);

    const windowShape = 2;
    const padding = 0;
    const dilationRate = 2;
    const strides: number = undefined;

    const result =
        tf.pool(x, windowShape, 'max', padding, dilationRate, strides);

    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(await result.data(), [11, 12, 15, 16]);
  });

  it('max x=[2,4,4,1] f=[2,2] s=1 d=2', async () => {
    // Feed forward.
    const x = tf.tensor4d(
        [
          1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
          1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 11, 13, 14, 16, 15
        ],
        [2, 4, 4, 1]);

    const windowShape = 2;
    const padding = 0;
    const dilationRate = 2;
    const strides: number = undefined;

    const result =
        tf.pool(x, windowShape, 'max', padding, dilationRate, strides);

    expect(result.shape).toEqual([2, 2, 2, 1]);
    expectArraysClose(await result.data(), [11, 12, 15, 16, 12, 11, 16, 15]);
  });

  it('avg x=[4,4,1] f=[2,2] s=1 d=2', async () => {
    // Feed forward.
    const a = tf.tensor3d(
        [1, 3, 2, 4, 6, 5, 8, 7, 9, 10, 12, 11, 16, 15, 14, 13], [4, 4, 1]);

    const windowShape = 2;
    const padding = 0;
    const dilationRate = 2;
    const strides: number = undefined;

    const result =
        tf.pool(a, windowShape, 'avg', padding, dilationRate, strides);

    expect(result.shape).toEqual([2, 2, 1]);
    expect(result.dtype).toBe('float32');
    expectArraysClose(await result.data(), [6, 7, 11, 10]);
  });

  it('max throws when neither s=1 nor d=1', () => {
    // Feed forward.
    const x = tf.tensor3d(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [4, 4, 1]);

    const windowShape = 2;
    const padding = 0;
    const dilationRate = 2;
    const strides = 2;

    expect(() => tf.pool(x, windowShape, 'max', padding, dilationRate, strides))
        .toThrowError();
  });

  it('avg throws when neither s=1 nor d=1', () => {
    // Feed forward.
    const x = tf.tensor3d(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [4, 4, 1]);

    const windowShape = 2;
    const padding = 0;
    const dilationRate = 2;
    const strides = 2;

    expect(() => tf.pool(x, windowShape, 'avg', padding, dilationRate, strides))
        .toThrowError();
  });
});

describeWithFlags('poolBackprop', ALL_ENVS, () => {
  it('max gradients x=[3,3,1] f=[2,2] s=1 d=1 no dup max value', async () => {
    const dy = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3, 1]);
    const expected = [0, 0, 0, 0, 1, 2, 0, 3, 4];

    const windowShape = 2;
    const padding = 0;
    const dilationRate: number = undefined;
    const strides: number = undefined;

    const dx = tf.grad(
        (x: tf.Tensor3D) =>
            x.pool(windowShape, 'max', padding, dilationRate, strides))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('max gradients x=[3,3,1] f=[2,2] s=1 d=2 no dup max value, test #1',
     async () => {
       const dy = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
       const x = tf.tensor3d(
           [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [4, 4, 1]);
       const expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4];

       const windowShape = 2;
       const padding = 0;
       const dilationRate = 2;
       const strides: number = undefined;

       const dx = tf.grad(
           (x: tf.Tensor3D) => x.pool(
               windowShape, 'max', padding, dilationRate, strides))(x, dy);

       expect(dx.shape).toEqual(x.shape);
       expectArraysClose(await dx.data(), expected);
     });

  it('max gradients x=[3,3,1] f=[2,2] s=1 d=2 no dup max value, test #2',
     async () => {
       const dy = tf.tensor3d([1, 2, 3, 4], [2, 2, 1]);
       const x = tf.tensor3d(
           [9, 5, 8, 6, 3, 1, 2, 4, 7, 3, 6, 4, 11, 15, 10, 16], [4, 4, 1]);
       const expected = [1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 4];

       const windowShape = 2;
       const padding = 0;
       const dilationRate = 2;
       const strides: number = undefined;

       const dx = tf.grad(
           (x: tf.Tensor3D) => x.pool(
               windowShape, 'max', padding, dilationRate, strides))(x, dy);

       expect(dx.shape).toEqual(x.shape);
       expectArraysClose(await dx.data(), expected);
     });

  it('max gradient x=[3,3,1] f=[2,2] s=1 d=2 dup max value', async () => {
    const dy = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3, 1]);
    const x = tf.tensor3d(
        [
          0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ],
        [5, 5, 1]);
    const expected = [
      0, 0, 0, 0, 0, 0, 5, 10, 0, 0, 0, 10, 20,
      0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0
    ];

    const windowShape = 2;
    const padding = 0;
    const dilationRate = 2;
    const strides: number = undefined;

    const dx = tf.grad(
        (x: tf.Tensor3D) =>
            x.pool(windowShape, 'max', padding, dilationRate, strides))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('avg gradient x=[4,4,1] f=[2,2] s=1 d=2', async () => {
    const x = tf.tensor3d(
        [
          1,  3,  2,  4,  6,  5,  8,  7,  9,  10, 12, 11, 16,
          15, 14, 13, 17, 18, 19, 20, 21, 22, 23, 24, 25
        ],
        [5, 5, 1]);
    const dy = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3, 1]);
    const f = 1 / (2 * 2);

    const windowShape = 2;
    const padding = 0;
    const dilationRate = 2;
    const strides: number = undefined;

    const dx = tf.grad(
        (x: tf.Tensor3D) =>
            x.pool(windowShape, 'avg', padding, dilationRate, strides))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), [
      1 * f, 2 * f, 4 * f,  2 * f,  3 * f,  4 * f,  5 * f, 10 * f, 5 * f,
      6 * f, 8 * f, 10 * f, 20 * f, 10 * f, 12 * f, 4 * f, 5 * f,  10 * f,
      5 * f, 6 * f, 7 * f,  8 * f,  16 * f, 8 * f,  9 * f
    ]);
  });
});

describeWithFlags('maxPool3d', ALL_ENVS, () => {
  it('4D x=[2,2,2,1] f=[2,2,2] s=1 p=valid', async () => {
    const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);

    const result = tf.maxPool3d(x, 2, 1, 'valid');

    expect(result.shape).toEqual([1, 1, 1, 1]);
    expectArraysClose(await result.data(), [8]);
  });

  it('x=[1,1,1,1,1] f=[1,1,1] s=1 [0] => [0]', async () => {
    const x = tf.tensor5d([0], [1, 1, 1, 1, 1]);

    const result = tf.maxPool3d(x, 1, 1, 0);

    expect(result.shape).toEqual([1, 1, 1, 1, 1]);
    expectArraysClose(await result.data(), [0]);
  });

  it('x=[1,2,2,2,1] f=[2,2,2] s=1 p=valid', async () => {
    const x = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);

    const result = tf.maxPool3d(x, 2, 1, 'valid');

    expect(result.shape).toEqual([1, 1, 1, 1, 1]);
    expectArraysClose(await result.data(), [8]);
  });

  it('x=[1,2,2,2,1] f=[2,2,2] s=1 p=same', async () => {
    const x = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
    const expected = [8, 8, 8, 8, 8, 8, 8, 8];
    const result = tf.maxPool3d(x, 2, 1, 'same');

    expect(result.shape).toEqual([1, 2, 2, 2, 1]);
    expectArraysClose(await result.data(), expected);
  });

  it('x=[1,3,3,3,1] f=[2,2,2] s=1 p=valid', async () => {
    const x = tf.tensor5d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27
        ],
        [1, 3, 3, 3, 1]);
    const expected = [14, 15, 17, 18, 23, 24, 26, 27];
    const result = tf.maxPool3d(x, 2, 1, 'valid');

    expect(result.shape).toEqual([1, 2, 2, 2, 1]);
    expectArraysClose(await result.data(), expected);
  });

  it('x=[1,3,3,3,1] f=[2,2,2] s=1 p=same', async () => {
    const x = tf.tensor5d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27
        ],
        [1, 3, 3, 3, 1]);
    const expected = [
      14, 15, 15, 17, 18, 18, 17, 18, 18, 23, 24, 24, 26, 27,
      27, 26, 27, 27, 23, 24, 24, 26, 27, 27, 26, 27, 27
    ];
    const result = tf.maxPool3d(x, 2, 1, 'same');

    expect(result.shape).toEqual([1, 3, 3, 3, 1]);
    expectArraysClose(await result.data(), expected);
  });

  it('x=[1,3,3,3,1] f=[2,2,2] s=1 p=valid, ignores NaNs', async () => {
    const x = tf.tensor5d(
        [
          1,  2,  3,  4,  5,  6,  NaN, 8,   NaN, 10, 11, 12, 13, 14,
          15, 16, 17, 18, 19, 20, 21,  NaN, 23,  24, 25, 26, 27
        ],
        [1, 3, 3, 3, 1]);
    const expected = [14, 15, 17, 18, 23, 24, 26, 27];
    const result = tf.maxPool3d(x, 2, 1, 'valid');

    expect(result.shape).toEqual([1, 2, 2, 2, 1]);
    expectArraysClose(await result.data(), expected);
  });

  it('x=[2,3,3,3,1] f=[2,2,2] s=1 p=valid', async () => {
    const x = tf.tensor5d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
          29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
          43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54
        ],
        [2, 3, 3, 3, 1]);
    const expected =
        [14, 15, 17, 18, 23, 24, 26, 27, 41, 42, 44, 45, 50, 51, 53, 54];
    const result = tf.maxPool3d(x, 2, 1, 'valid');

    expect(result.shape).toEqual([2, 2, 2, 2, 1]);
    expectArraysClose(await result.data(), expected);
  });

  it('x=[1,3,3,3,2] f=[2,2,2] s=1 p=valid', async () => {
    const x = tf.tensor5d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
          29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
          43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54
        ],
        [1, 3, 3, 3, 2]);
    const expected =
        [27, 28, 29, 30, 33, 34, 35, 36, 45, 46, 47, 48, 51, 52, 53, 54];
    const result = tf.maxPool3d(x, 2, 1, 'valid');

    expect(result.shape).toEqual([1, 2, 2, 2, 2]);
    expectArraysClose(await result.data(), expected);
  });

  it('x=[1,2,2,2,1] f=[2,2,2] s=1 p=1 roundingMode=floor', async () => {
    const x = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
    const expected = [
      1, 2, 2, 3, 4, 4, 3, 4, 4, 5, 6, 6, 7, 8,
      8, 7, 8, 8, 5, 6, 6, 7, 8, 8, 7, 8, 8
    ];
    const result = tf.maxPool3d(x, 2, 1, 1, 'floor');

    expect(result.shape).toEqual([1, 3, 3, 3, 1]);
    expectArraysClose(await result.data(), expected);
  });

  it('throws when x is not rank 5', async () => {
    // tslint:disable-next-line:no-any
    const x: any = tf.tensor1d([1]);

    expect(() => tf.maxPool3d(x as tf.Tensor5D, 2, 1, 'valid')).toThrowError();
  });

  it('throws when dimRoundingMode is set and pad is not a number', async () => {
    const x = tf.tensor5d([1], [1, 1, 1, 1, 1]);
    const pad = 'valid';
    const dimRoundingMode = 'round';

    expect(() => tf.maxPool3d(x, 2, 1, pad, dimRoundingMode)).toThrowError();
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.maxPool3d({} as tf.Tensor5D, 2, 1, 'valid')).toThrowError();
  });

  it('accepts a tensor-like object', async () => {
    const x = [[[[[0]]]]];  // 1x1x1x1x1
    const result = tf.maxPool3d(x, 1, 1, 0);
    expect(result.shape).toEqual([1, 1, 1, 1, 1]);
    expectArraysClose(await result.data(), [0]);
  });
});

describeWithFlags('maxPool3dBackprop', ALL_ENVS, () => {
  it('gradient x=[2,2,2,1] f=[1,1,1] s=1', async () => {
    const dy = tf.tensor4d([1, 2, 1, 2, 1, 2, 1, 2], [2, 2, 2, 1]);
    const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);
    const expected = [1, 2, 1, 2, 1, 2, 1, 2];

    const dx = tf.grad((x: tf.Tensor4D) => tf.maxPool3d(x, 1, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[1,2,2,2,1] f=[1,1,1] s=1', async () => {
    const dy = tf.tensor5d([1, 2, 1, 2, 1, 2, 1, 2], [1, 2, 2, 2, 1]);
    const x = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
    const expected = [1, 2, 1, 2, 1, 2, 1, 2];

    const dx = tf.grad((x: tf.Tensor5D) => tf.maxPool3d(x, 1, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[1,2,2,2,1] f=[2,2,2] s=2', async () => {
    const dy = tf.tensor5d([1], [1, 1, 1, 1, 1]);
    const x = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
    const expected = [0, 0, 0, 0, 0, 0, 0, 1];

    const dx = tf.grad((x: tf.Tensor5D) => tf.maxPool3d(x, 2, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient with clone x=[1,2,2,2,1] f=[2,2,2] s=1', async () => {
    const dy = tf.tensor5d([1], [1, 1, 1, 1, 1]);
    const x = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
    const expected = [0, 0, 0, 0, 0, 0, 0, 1];

    const dx = tf.grad(
        (x: tf.Tensor5D) => tf.maxPool3d(x.clone(), 2, 1, 0).clone())(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[1,3,3,3,1] f=[2,2,2] s=1 no dup max value', async () => {
    const dy = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
    const x = tf.tensor5d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27
        ],
        [1, 3, 3, 3, 1]);
    const expected = [
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      2, 0, 3, 4, 0, 0, 0, 0, 5, 6, 0, 7, 8
    ];

    const dx = tf.grad((x: tf.Tensor5D) => tf.maxPool3d(x, 2, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[1,3,3,3,1] f=[2,2,2] s=1 dup max value', async () => {
    const dy = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
    const x = tf.tensor5d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 27,
          15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 14
        ],
        [1, 3, 3, 3, 1]);
    const expected = [
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ];

    const dx = tf.grad((x: tf.Tensor5D) => tf.maxPool3d(x, 2, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[1,4,4,4,1] f=[2,2,2] s=2', async () => {
    const dy = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
    const x = tf.tensor5d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
          17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
          33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
          49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
        ],
        [1, 4, 4, 4, 1]);
    const expected = [
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 2, 0, 0, 0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 6, 0, 0, 0, 0, 0, 7, 0, 8
    ];

    const dx = tf.grad((x: tf.Tensor5D) => tf.maxPool3d(x, 2, 2, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[1,3,3,3,2] f=[2,2,2] s=1 no dup max value', async () => {
    const dy = tf.tensor5d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [1, 2, 2, 2, 2]);
    const x = tf.tensor5d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
          29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
          43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54
        ],
        [1, 3, 3, 3, 2]);
    const expected = [
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0, 0, 0,  0,  0,  0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 2,  3,  4,  0, 0, 5,  6,  7,  8,
      0, 0, 0, 0, 0, 0, 0, 0, 9, 10, 11, 12, 0, 0, 13, 14, 15, 16
    ];

    const dx = tf.grad((x: tf.Tensor5D) => tf.maxPool3d(x, 2, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[1,3,3,3,2] f=[2,2,2] s=1 dup max value', async () => {
    const dy = tf.tensor5d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [1, 2, 2, 2, 2]);
    const x = tf.tensor5d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 53, 54,
          29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
          43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 27, 28
        ],
        [1, 3, 3, 3, 2]);
    const expected = [
      0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 64, 72, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0
    ];

    const dx = tf.grad((x: tf.Tensor5D) => tf.maxPool3d(x, 2, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[2,3,3,3,1] f=[2,2,2] s=1 no dup max value', async () => {
    const dy = tf.tensor5d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [2, 2, 2, 2, 1]);
    const x = tf.tensor5d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
          29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
          43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54
        ],
        [2, 3, 3, 3, 1]);
    const expected = [
      0, 0, 0, 0, 0, 0,  0, 0,  0,  0, 0, 0, 0, 1,  2,  0, 3,  4,
      0, 0, 0, 0, 5, 6,  0, 7,  8,  0, 0, 0, 0, 0,  0,  0, 0,  0,
      0, 0, 0, 0, 9, 10, 0, 11, 12, 0, 0, 0, 0, 13, 14, 0, 15, 16
    ];

    const dx = tf.grad((x: tf.Tensor5D) => tf.maxPool3d(x, 2, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[2,3,3,3,1] f=[2,2,2] s=1 dup max value', async () => {
    const dy = tf.tensor5d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [2, 2, 2, 2, 1]);
    const x = tf.tensor5d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 27,
          15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 14, 28,
          29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 54, 42,
          43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 41
        ],
        [2, 3, 3, 3, 1]);
    const expected = [
      0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 0,
      0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,
      0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0
    ];

    const dx = tf.grad((x: tf.Tensor5D) => tf.maxPool3d(x, 2, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });
});

describeWithFlags('avgPool3d', ALL_ENVS, () => {
  it('x=[2,2,2,1] f=[2,2,2] s=1 p=valid', async () => {
    const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);

    const result = tf.avgPool3d(x, 2, 1, 'valid');

    expect(result.shape).toEqual([1, 1, 1, 1]);
    expectArraysClose(await result.data(), [4.5]);
  });

  it('x=[1,1,1,1,1] f=[1,1,1] s=1 [0] => [0]', async () => {
    const x = tf.tensor5d([0], [1, 1, 1, 1, 1]);

    const result = tf.avgPool3d(x, 1, 1, 0);

    expect(result.shape).toEqual([1, 1, 1, 1, 1]);
    expectArraysClose(await result.data(), [0]);
  });

  it('x=[1,2,2,2,1] f=[2,2,2] s=1 p=valid', async () => {
    const x = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);

    const result = tf.avgPool3d(x, 2, 1, 'valid');

    expect(result.shape).toEqual([1, 1, 1, 1, 1]);
    expectArraysClose(await result.data(), [4.5]);
  });

  it('x=[1,2,2,2,1] f=[2,2,2] s=1 p=same', async () => {
    const x = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
    const expected = [4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8];
    const result = tf.avgPool3d(x, 2, 1, 'same');

    expect(result.shape).toEqual([1, 2, 2, 2, 1]);
    expectArraysClose(await result.data(), expected);
  });

  it('x=[1,3,3,3,1] f=[2,2,2] s=1 p=valid', async () => {
    const x = tf.tensor5d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27
        ],
        [1, 3, 3, 3, 1]);
    const expected = [7.5, 8.5, 10.5, 11.5, 16.5, 17.5, 19.5, 20.5];
    const result = tf.avgPool3d(x, 2, 1, 'valid');

    expect(result.shape).toEqual([1, 2, 2, 2, 1]);
    expectArraysClose(await result.data(), expected);
  });

  it('x=[1,3,3,3,1] f=[2,2,2] s=1 p=same', async () => {
    const x = tf.tensor5d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27
        ],
        [1, 3, 3, 3, 1]);
    const expected = [
      7.5,  8.5,  9,    10.5, 11.5, 12,   12,   13,   13.5,
      16.5, 17.5, 18,   19.5, 20.5, 21,   21,   22,   22.5,
      21,   22,   22.5, 24,   25,   25.5, 25.5, 26.5, 27
    ];
    const result = tf.avgPool3d(x, 2, 1, 'same');

    expect(result.shape).toEqual([1, 3, 3, 3, 1]);
    expectArraysClose(await result.data(), expected);
  });

  it('x=[1,3,3,3,1] f=[2,2,2] s=1 p=valid, propagates NaNs', async () => {
    const x = tf.tensor5d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,   NaN, 10, 11, 12, 13, 14,
          15, 16, 17, 18, 19, 20, 21, NaN, 23,  24, 25, 26, 27
        ],
        [1, 3, 3, 3, 1]);
    const expected = [7.5, 8.5, 10.5, NaN, NaN, 17.5, NaN, 20.5];
    const result = tf.avgPool3d(x, 2, 1, 'valid');

    expect(result.shape).toEqual([1, 2, 2, 2, 1]);
    expectArraysClose(await result.data(), expected);
  });

  it('x=[2,3,3,3,1] f=[2,2,2] s=1 p=valid', async () => {
    const x = tf.tensor5d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
          29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
          43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54
        ],
        [2, 3, 3, 3, 1]);
    const expected = [
      7.5, 8.5, 10.5, 11.5, 16.5, 17.5, 19.5, 20.5, 34.5, 35.5, 37.5, 38.5,
      43.5, 44.5, 46.5, 47.5
    ];
    const result = tf.avgPool3d(x, 2, 1, 'valid');

    expect(result.shape).toEqual([2, 2, 2, 2, 1]);
    expectArraysClose(await result.data(), expected);
  });

  it('x=[1,3,3,3,2] f=[2,2,2] s=1 p=valid', async () => {
    const x = tf.tensor5d(
        [
          1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
          15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
          29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
          43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54
        ],
        [1, 3, 3, 3, 2]);
    const expected =
        [14, 15, 16, 17, 20, 21, 22, 23, 32, 33, 34, 35, 38, 39, 40, 41];
    const result = tf.avgPool3d(x, 2, 1, 'valid');

    expect(result.shape).toEqual([1, 2, 2, 2, 2]);
    expectArraysClose(await result.data(), expected);
  });

  it('x=[1,2,2,2,1] f=[2,2,2] s=1 p=1 roundingMode=floor', async () => {
    const x = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
    const expected = [
      1, 1.5, 2,   2, 2.5, 3,   3, 3.5, 4,   3, 3.5, 4,   4, 4.5,
      5, 5,   5.5, 6, 5,   5.5, 6, 6,   6.5, 7, 7,   7.5, 8
    ];
    const result = tf.avgPool3d(x, 2, 1, 1, 'floor');

    expect(result.shape).toEqual([1, 3, 3, 3, 1]);
    expectArraysClose(await result.data(), expected);
  });

  it('throws when x is not rank 5', async () => {
    // tslint:disable-next-line:no-any
    const x: any = tf.tensor1d([1]);

    expect(() => tf.avgPool3d(x as tf.Tensor5D, 2, 1, 'valid')).toThrowError();
  });

  it('throws when dimRoundingMode is set and pad is not a number', async () => {
    const x = tf.tensor5d([1], [1, 1, 1, 1, 1]);
    const pad = 'valid';
    const dimRoundingMode = 'round';

    expect(() => tf.avgPool3d(x, 2, 1, pad, dimRoundingMode)).toThrowError();
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.avgPool3d({} as tf.Tensor5D, 2, 1, 'valid')).toThrowError();
  });

  it('throws when input dtype is not float32', () => {
    const a = tf.tensor5d([1], [1, 1, 1, 1, 1], 'int32');
    expect(() => tf.avgPool3d(a, 2, 1, 0)).toThrowError();
  });

  it('accepts a tensor-like object', async () => {
    const x = [[[[[0]]]]];  // 1x1x1x1x1
    const result = tf.avgPool3d(x, 1, 1, 0);
    expect(result.shape).toEqual([1, 1, 1, 1, 1]);
    expectArraysClose(await result.data(), [0]);
  });
});

describeWithFlags('avgPool3dBackprop', ALL_ENVS, () => {
  it('gradient x=[2,2,2,1] f=[1,1,1] s=1', async () => {
    const dy = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);
    const x = tf.ones([2, 2, 2, 1]) as tf.Tensor4D;
    const expected = [1, 2, 3, 4, 5, 6, 7, 8];

    const dx = tf.grad((x: tf.Tensor4D) => tf.avgPool3d(x, 1, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[1,2,2,2,1] f=[1,1,1] s=1', async () => {
    const dy = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
    const x = tf.ones([1, 2, 2, 2, 1]) as tf.Tensor5D;
    const expected = [1, 2, 3, 4, 5, 6, 7, 8];

    const dx = tf.grad((x: tf.Tensor5D) => tf.avgPool3d(x, 1, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[1,2,2,2,1] f=[2,2,2] s=2', async () => {
    const dy = tf.tensor5d([8], [1, 1, 1, 1, 1]);
    const x = tf.ones([1, 2, 2, 2, 1]) as tf.Tensor5D;
    const expected = [1, 1, 1, 1, 1, 1, 1, 1];

    const dx = tf.grad((x: tf.Tensor5D) => tf.avgPool3d(x, 2, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient with clone x=[1,2,2,2,1] f=[2,2,2] s=1', async () => {
    const dy = tf.tensor5d([8], [1, 1, 1, 1, 1]);
    const x = tf.ones([1, 2, 2, 2, 1]) as tf.Tensor5D;
    const expected = [1, 1, 1, 1, 1, 1, 1, 1];

    const dx = tf.grad(
        (x: tf.Tensor5D) => tf.avgPool3d(x.clone(), 2, 1, 0).clone())(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[1,3,3,3,1] f=[2,2,2] s=1', async () => {
    const dy = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
    const x = tf.ones([1, 3, 3, 3, 1]) as tf.Tensor5D;
    const expected = [
      0.125, 0.375, 0.25, 0.5, 1.25, 0.75, 0.375, 0.875, 0.5,
      0.75,  1.75,  1,    2,   4.5,  2.5,  1.25,  2.75,  1.5,
      0.625, 1.375, 0.75, 1.5, 3.25, 1.75, 0.875, 1.875, 1
    ];

    const dx = tf.grad((x: tf.Tensor5D) => tf.avgPool3d(x, 2, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[1,4,4,4,1] f=[2,2,2] s=2', async () => {
    const dy = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
    const x = tf.ones([1, 4, 4, 4, 1]) as tf.Tensor5D;
    const expected = [
      0.125, 0.125, 0.25,  0.25,  0.125, 0.125, 0.25,  0.25,  0.375, 0.375,
      0.5,   0.5,   0.375, 0.375, 0.5,   0.5,   0.125, 0.125, 0.25,  0.25,
      0.125, 0.125, 0.25,  0.25,  0.375, 0.375, 0.5,   0.5,   0.375, 0.375,
      0.5,   0.5,   0.625, 0.625, 0.75,  0.75,  0.625, 0.625, 0.75,  0.75,
      0.875, 0.875, 1,     1,     0.875, 0.875, 1,     1,     0.625, 0.625,
      0.75,  0.75,  0.625, 0.625, 0.75,  0.75,  0.875, 0.875, 1,     1,
      0.875, 0.875, 1,     1
    ];

    const dx = tf.grad((x: tf.Tensor5D) => tf.avgPool3d(x, 2, 2, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[1,3,3,3,2] f=[2,2,2] s=1', async () => {
    const dy = tf.tensor5d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [1, 2, 2, 2, 2]);
    const x = tf.ones([1, 3, 3, 3, 2]) as tf.Tensor5D;
    const expected = [
      0.125, 0.25,  0.5,  0.75,  0.375, 0.5,   0.75, 1,     2,     2.5,  1.25,
      1.5,   0.625, 0.75, 1.5,   1.75,  0.875, 1,    1.25,  1.5,   3,    3.5,
      1.75,  2,     3.5,  4,     8,     9,     4.5,  5,     2.25,  2.5,  5,
      5.5,   2.75,  3,    1.125, 1.25,  2.5,   2.75, 1.375, 1.5,   2.75, 3,
      6,     6.5,   3.25, 3.5,   1.625, 1.75,  3.5,  3.75,  1.875, 2
    ];

    const dx = tf.grad((x: tf.Tensor5D) => tf.avgPool3d(x, 2, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[2,3,3,3,1] f=[2,2,2] s=1', async () => {
    const dy = tf.tensor5d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [2, 2, 2, 2, 1]);
    const x = tf.ones([2, 3, 3, 3, 1]) as tf.Tensor5D;
    const expected = [
      0.125, 0.375, 0.25,  0.5,   1.25, 0.75,  0.375, 0.875, 0.5,   0.75, 1.75,
      1,     2,     4.5,   2.5,   1.25, 2.75,  1.5,   0.625, 1.375, 0.75, 1.5,
      3.25,  1.75,  0.875, 1.875, 1,    1.125, 2.375, 1.25,  2.5,   5.25, 2.75,
      1.375, 2.875, 1.5,   2.75,  5.75, 3,     6,     12.5,  6.5,   3.25, 6.75,
      3.5,   1.625, 3.375, 1.75,  3.5,  7.25,  3.75,  1.875, 3.875, 2
    ];

    const dx = tf.grad((x: tf.Tensor5D) => tf.avgPool3d(x, 2, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });
});