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
