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
