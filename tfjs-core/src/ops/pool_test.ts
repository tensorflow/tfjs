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
import {expectArraysClose} from '../test_util';

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

  it('max x=[2,2,3] f=[1,1] s=2 p=1 fractional outputs default rounding',
    async () => {
      // Feed forward.
      const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 2, 3]);

      const windowShape = 1;
      const padding = 1;
      const dilationRate: number = undefined;
      const strides = 2;

      const result =
          tf.pool(a, windowShape, 'max', padding, dilationRate, strides);
      expect(result.shape).toEqual([2, 2, 3]);
      expectArraysClose(
        await result.data(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
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

  it('avg x=[2,2,3] f=[1,1] s=2 p=1 fractional outputs default rounding',
    async () => {
      // Feed forward.
      const a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 2, 3]);

      const windowShape = 1;
      const padding = 1;
      const dilationRate: number = undefined;
      const strides = 2;

      const result =
          tf.pool(a, windowShape, 'avg', padding, dilationRate, strides);
      expect(result.shape).toEqual([2, 2, 3]);
      expectArraysClose(
        await result.data(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
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
