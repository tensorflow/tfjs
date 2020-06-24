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
    const x: tf.Tensor4D = tf.ones([2, 2, 2, 1]);
    const expected = [1, 2, 3, 4, 5, 6, 7, 8];

    const dx = tf.grad((x: tf.Tensor4D) => tf.avgPool3d(x, 1, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[1,2,2,2,1] f=[1,1,1] s=1', async () => {
    const dy = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
    const x: tf.Tensor5D = tf.ones([1, 2, 2, 2, 1]);
    const expected = [1, 2, 3, 4, 5, 6, 7, 8];

    const dx = tf.grad((x: tf.Tensor5D) => tf.avgPool3d(x, 1, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[1,2,2,2,1] f=[2,2,2] s=2', async () => {
    const dy = tf.tensor5d([8], [1, 1, 1, 1, 1]);
    const x: tf.Tensor5D = tf.ones([1, 2, 2, 2, 1]);
    const expected = [1, 1, 1, 1, 1, 1, 1, 1];

    const dx = tf.grad((x: tf.Tensor5D) => tf.avgPool3d(x, 2, 1, 0))(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient with clone x=[1,2,2,2,1] f=[2,2,2] s=1', async () => {
    const dy = tf.tensor5d([8], [1, 1, 1, 1, 1]);
    const x: tf.Tensor5D = tf.ones([1, 2, 2, 2, 1]);
    const expected = [1, 1, 1, 1, 1, 1, 1, 1];

    const dx = tf.grad(
        (x: tf.Tensor5D) => tf.avgPool3d(x.clone(), 2, 1, 0).clone())(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), expected);
  });

  it('gradient x=[1,3,3,3,1] f=[2,2,2] s=1', async () => {
    const dy = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
    const x: tf.Tensor5D = tf.ones([1, 3, 3, 3, 1]);
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
    const x: tf.Tensor5D = tf.ones([1, 4, 4, 4, 1]);
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
    const x: tf.Tensor5D = tf.ones([1, 3, 3, 3, 2]);
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
    const x: tf.Tensor5D = tf.ones([2, 3, 3, 3, 1]);
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
