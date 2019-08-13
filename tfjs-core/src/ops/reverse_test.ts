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

describeWithFlags('reverse1d', ALL_ENVS, () => {
  it('reverse a 1D array', async () => {
    const input = tf.tensor1d([1, 2, 3, 4, 5]);
    const result = tf.reverse1d(input);
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(await result.data(), [5, 4, 3, 2, 1]);
  });

  it('reverse a 1D array, even length', async () => {
    const input = tf.tensor1d([1, 2, 3, 4]);
    const result = tf.reverse1d(input);
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(await result.data(), [4, 3, 2, 1]);
  });

  it('grad', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const dy = tf.tensor1d([10, 20, 30]);
    const da = tf.grad((a: tf.Tensor1D) => tf.reverse1d(a))(a, dy);
    expect(da.shape).toEqual([3]);
    expectArraysClose(await da.data(), [30, 20, 10]);
  });

  it('gradient with clones', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const dy = tf.tensor1d([10, 20, 30]);
    const da =
        tf.grad((a: tf.Tensor1D) => tf.reverse1d(a.clone()).clone())(a, dy);
    expect(da.shape).toEqual([3]);
    expectArraysClose(await da.data(), [30, 20, 10]);
  });

  it('accepts a tensor-like object', async () => {
    const input = [1, 2, 3, 4, 5];
    const result = tf.reverse1d(input);
    expect(result.shape).toEqual([5]);
    expectArraysClose(await result.data(), [5, 4, 3, 2, 1]);
  });
});

describeWithFlags('reverse2d', ALL_ENVS, () => {
  it('reverse a 2D array at axis [0]', async () => {
    const axis = [0];
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const result = tf.reverse2d(a, axis);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(await result.data(), [4, 5, 6, 1, 2, 3]);
  });

  it('reverse a 2D array at axis [1]', async () => {
    const axis = [1];
    const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
    const result = tf.reverse2d(a, axis);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(await result.data(), [3, 2, 1, 6, 5, 4]);
  });

  it('reverse a 2D array odd rows and columns at axis [0, 1]', async () => {
    const axis = [0, 1];
    const a = tf.tensor2d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [3, 5]);
    const result = tf.reverse2d(a, axis);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(
        await result.data(),
        [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
  });

  it('throws error with invalid input', () => {
    // tslint:disable-next-line:no-any
    const x: any = tf.tensor1d([1, 20, 300, 4]);
    expect(() => tf.reverse2d(x, [0])).toThrowError();
  });

  it('throws error with invalid axis param', () => {
    const x = tf.tensor2d([1, 20, 300, 4], [1, 4]);
    expect(() => tf.reverse2d(x, [2])).toThrowError();
    expect(() => tf.reverse2d(x, [-3])).toThrowError();
  });

  it('throws error with non integer axis param', () => {
    const x = tf.tensor2d([1, 20, 300, 4], [1, 4]);
    expect(() => tf.reverse2d(x, [0.5])).toThrowError();
  });

  it('grad', async () => {
    const a = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
    const dy = tf.tensor2d([[10, 20, 30], [40, 50, 60]]);
    const da = tf.grad((a: tf.Tensor2D) => tf.reverse2d(a))(a, dy);
    expect(da.shape).toEqual([2, 3]);
    expectArraysClose(await da.data(), [60, 50, 40, 30, 20, 10]);
  });

  it('grad with reverse(axis=0)', async () => {
    const a = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
    const dy = tf.tensor2d([[10, 20, 30], [40, 50, 60]]);
    const da = tf.grad((a: tf.Tensor2D) => tf.reverse2d(a, 0))(a, dy);
    expect(da.shape).toEqual([2, 3]);
    expectArraysClose(await da.data(), [40, 50, 60, 10, 20, 30]);
  });

  it('grad with reverse(axis=1)', async () => {
    const a = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
    const dy = tf.tensor2d([[10, 20, 30], [40, 50, 60]]);
    const da = tf.grad((a: tf.Tensor2D) => tf.reverse2d(a, 1))(a, dy);
    expect(da.shape).toEqual([2, 3]);
    expectArraysClose(await da.data(), [30, 20, 10, 60, 50, 40]);
  });

  it('accepts a tensor-like object', async () => {
    const axis = [0];
    const a = [[1, 2, 3], [4, 5, 6]];  // 2x3
    const result = tf.reverse2d(a, axis);

    expect(result.shape).toEqual([2, 3]);
    expectArraysClose(await result.data(), [4, 5, 6, 1, 2, 3]);
  });
});

describeWithFlags('reverse3d', ALL_ENVS, () => {
  // [
  //   [
  //     [0,  1,  2,  3],
  //     [4,  5,  6,  7],
  //     [8,  9,  10, 11]
  //   ],
  //   [
  //     [12, 13, 14, 15],
  //     [16, 17, 18, 19],
  //     [20, 21, 22, 23]
  //   ]
  // ]
  const shape: [number, number, number] = [2, 3, 4];
  const data = [
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
  ];

  it('reverse a 3D array at axis [0]', async () => {
    const input = tf.tensor3d(data, shape);
    const result = tf.reverse3d(input, [0]);
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(await result.data(), [
      12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11
    ]);
  });

  it('reverse a 3D array at axis [1]', async () => {
    const input = tf.tensor3d(data, shape);
    const result = tf.reverse3d(input, [1]);
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(await result.data(), [
      8,  9,  10, 11, 4,  5,  6,  7,  0,  1,  2,  3,
      20, 21, 22, 23, 16, 17, 18, 19, 12, 13, 14, 15
    ]);
  });

  it('reverse a 3D array at axis [2]', async () => {
    const input = tf.tensor3d(data, shape);
    const result = tf.reverse3d(input, [2]);
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(await result.data(), [
      3,  2,  1,  0,  7,  6,  5,  4,  11, 10, 9,  8,
      15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20
    ]);
  });

  it('reverse a 3D array at axis [0, 1]', async () => {
    const input = tf.tensor3d(data, shape);
    const result = tf.reverse3d(input, [0, 1]);
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(await result.data(), [
      20, 21, 22, 23, 16, 17, 18, 19, 12, 13, 14, 15,
      8,  9,  10, 11, 4,  5,  6,  7,  0,  1,  2,  3
    ]);
  });

  it('reverse a 3D array at axis [0, 2]', async () => {
    const input = tf.tensor3d(data, shape);
    const result = tf.reverse3d(input, [0, 2]);
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(await result.data(), [
      15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20,
      3,  2,  1,  0,  7,  6,  5,  4,  11, 10, 9,  8
    ]);
  });

  it('reverse a 3D array at axis [1, 2]', async () => {
    const input = tf.tensor3d(data, shape);
    const result = tf.reverse3d(input, [1, 2]);
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(await result.data(), [
      11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0,
      23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12
    ]);
  });

  it('throws error with invalid input', () => {
    // tslint:disable-next-line:no-any
    const x: any = tf.tensor2d([1, 20, 300, 4], [1, 4]);
    expect(() => tf.reverse3d(x, [1])).toThrowError();
  });

  it('throws error with invalid axis param', () => {
    const x = tf.tensor3d([1, 20, 300, 4], [1, 1, 4]);
    expect(() => tf.reverse3d(x, [3])).toThrowError();
    expect(() => tf.reverse3d(x, [-4])).toThrowError();
  });

  it('throws error with non integer axis param', () => {
    const x = tf.tensor3d([1, 20, 300, 4], [1, 1, 4]);
    expect(() => tf.reverse3d(x, [0.5])).toThrowError();
  });

  it('accepts a tensor-like object', async () => {
    const input = [[[1], [2], [3]], [[4], [5], [6]]];  // 2x3x1
    const result = tf.reverse3d(input, [0]);
    expect(result.shape).toEqual([2, 3, 1]);
    expectArraysClose(await result.data(), [4, 5, 6, 1, 2, 3]);
  });
});

describeWithFlags('reverse4d', ALL_ENVS, () => {
  // [
  //   [
  //     [
  //       [0,  1,  2,  3],
  //       [4,  5,  6,  7],
  //       [8,  9,  10, 11]
  //     ],
  //     [
  //       [12, 13, 14, 15],
  //       [16, 17, 18, 19],
  //       [20, 21, 22, 23]
  //     ]
  //   ],
  //   [
  //     [
  //       [24, 25, 26, 27],
  //       [28, 29, 30, 31],
  //       [32, 33, 34, 35]
  //     ],
  //     [
  //       [36, 37, 38, 39],
  //       [40, 41, 42, 43],
  //       [44, 45, 46, 47]
  //     ]
  //   ],
  //   [
  //     [
  //       [48, 49, 50, 51],
  //       [52, 53, 54, 55],
  //       [56, 57, 58, 59]
  //     ],
  //     [
  //       [60, 61, 62, 63],
  //       [64, 65, 66, 67],
  //       [68, 69, 70, 71]
  //     ]
  //   ]
  // ]
  const shape: [number, number, number, number] = [3, 2, 3, 4];
  const data = [
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71
  ];

  it('reverse a 4D array at axis [0]', async () => {
    const input = tf.tensor4d(data, shape);
    const result = tf.reverse4d(input, [0]);
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(await result.data(), [
      48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
      66, 67, 68, 69, 70, 71, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
      36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 0,  1,  2,  3,  4,  5,
      6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
    ]);
  });

  it('reverse a 4D array at axis [1]', async () => {
    const input = tf.tensor4d(data, shape);
    const result = tf.reverse4d(input, [1]);
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(await result.data(), [
      12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0,  1,  2,  3,  4,  5,
      6,  7,  8,  9,  10, 11, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
      24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 60, 61, 62, 63, 64, 65,
      66, 67, 68, 69, 70, 71, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
    ]);
  });

  it('reverse a 4D array at axis [2]', async () => {
    const input = tf.tensor4d(data, shape);
    const result = tf.reverse4d(input, [2]);
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(await result.data(), [
      8,  9,  10, 11, 4,  5,  6,  7,  0,  1,  2,  3,  20, 21, 22, 23, 16, 17,
      18, 19, 12, 13, 14, 15, 32, 33, 34, 35, 28, 29, 30, 31, 24, 25, 26, 27,
      44, 45, 46, 47, 40, 41, 42, 43, 36, 37, 38, 39, 56, 57, 58, 59, 52, 53,
      54, 55, 48, 49, 50, 51, 68, 69, 70, 71, 64, 65, 66, 67, 60, 61, 62, 63
    ]);
  });

  it('reverse a 4D array at axis [3]', async () => {
    const input = tf.tensor4d(data, shape);
    const result = tf.reverse4d(input, [3]);
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(await result.data(), [
      3,  2,  1,  0,  7,  6,  5,  4,  11, 10, 9,  8,  15, 14, 13, 12, 19, 18,
      17, 16, 23, 22, 21, 20, 27, 26, 25, 24, 31, 30, 29, 28, 35, 34, 33, 32,
      39, 38, 37, 36, 43, 42, 41, 40, 47, 46, 45, 44, 51, 50, 49, 48, 55, 54,
      53, 52, 59, 58, 57, 56, 63, 62, 61, 60, 67, 66, 65, 64, 71, 70, 69, 68
    ]);
  });

  it('reverse a 4D array at axis [0, 2]', async () => {
    const input = tf.tensor4d(data, shape);
    const result = tf.reverse4d(input, [0, 2]);
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(await result.data(), [
      56, 57, 58, 59, 52, 53, 54, 55, 48, 49, 50, 51, 68, 69, 70, 71, 64, 65,
      66, 67, 60, 61, 62, 63, 32, 33, 34, 35, 28, 29, 30, 31, 24, 25, 26, 27,
      44, 45, 46, 47, 40, 41, 42, 43, 36, 37, 38, 39, 8,  9,  10, 11, 4,  5,
      6,  7,  0,  1,  2,  3,  20, 21, 22, 23, 16, 17, 18, 19, 12, 13, 14, 15
    ]);
  });

  it('reverse a 4D array at axis [1, 3]', async () => {
    const input = tf.tensor4d(data, shape);
    const result = tf.reverse4d(input, [1, 3]);
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(await result.data(), [
      15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20, 3,  2,  1,  0,  7,  6,
      5,  4,  11, 10, 9,  8,  39, 38, 37, 36, 43, 42, 41, 40, 47, 46, 45, 44,
      27, 26, 25, 24, 31, 30, 29, 28, 35, 34, 33, 32, 63, 62, 61, 60, 67, 66,
      65, 64, 71, 70, 69, 68, 51, 50, 49, 48, 55, 54, 53, 52, 59, 58, 57, 56
    ]);
  });

  it('throws error with invalid input', () => {
    // tslint:disable-next-line:no-any
    const x: any = tf.tensor3d([1, 20, 300, 4], [1, 1, 4]);
    expect(() => tf.reverse4d(x, [1])).toThrowError();
  });

  it('throws error with invalid axis param', () => {
    const x = tf.tensor4d([1, 20, 300, 4], [1, 1, 1, 4]);
    expect(() => tf.reverse4d(x, [4])).toThrowError();
    expect(() => tf.reverse4d(x, [-5])).toThrowError();
  });

  it('throws error with non integer axis param', () => {
    const x = tf.tensor4d([1, 20, 300, 4], [1, 1, 1, 4]);
    expect(() => tf.reverse4d(x, [0.5])).toThrowError();
  });

  it('accepts a tensor-like object', async () => {
    const input = [[[[1]], [[2]], [[3]]], [[[4]], [[5]], [[6]]]];  // 2x3x1x1
    const result = tf.reverse4d(input, [0]);
    expect(result.shape).toEqual([2, 3, 1, 1]);
    expectArraysClose(await result.data(), [4, 5, 6, 1, 2, 3]);
  });
});

describeWithFlags('reverse', ALL_ENVS, () => {
  it('throws when passed a non-tensor', () => {
    expect(() => tf.reverse({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'reverse' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const input = [1, 2, 3];
    const result = tf.reverse(input);
    expect(result.shape).toEqual([3]);
    expectArraysClose(await result.data(), [3, 2, 1]);
  });
});
