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
