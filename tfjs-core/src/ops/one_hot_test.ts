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

describeWithFlags('oneHot', ALL_ENVS, () => {
  it('Depth 1 throws error', () => {
    const indices = tf.tensor1d([0, 0, 0], 'int32');
    expect(() => tf.oneHot(indices, 1)).toThrowError();
  });

  it('Depth 2, diagonal', async () => {
    const indices = tf.tensor1d([0, 1], 'int32');
    const res = tf.oneHot(indices, 2);

    expect(res.shape).toEqual([2, 2]);
    expectArraysClose(await res.data(), [1, 0, 0, 1]);
  });

  it('Scalar input as Tensor', async () => {
    const indices = tf.scalar(2, 'int32');
    const res = tf.oneHot(indices, 4);

    expect(res.shape).toEqual([4]);
    expectArraysClose(await res.data(), [0, 0, 1, 0]);
  });

  it('Scalar input as number', async () => {
    const indices = 2;
    const res = tf.oneHot(indices, 4);

    expect(res.shape).toEqual([4]);
    expectArraysClose(await res.data(), [0, 0, 1, 0]);
  });

  it('oneHot with chaining compiles', () => {
    const indices = 2;
    // Asserts that there is no compiler error.
    tf.oneHot(indices, 4).toFloat();
  });

  it('Depth 2, transposed diagonal', async () => {
    const indices = tf.tensor1d([1, 0], 'int32');
    const res = tf.oneHot(indices, 2);

    expect(res.shape).toEqual([2, 2]);
    expectArraysClose(await res.data(), [0, 1, 1, 0]);
  });

  it('Depth 3, 4 events', async () => {
    const indices = tf.tensor1d([2, 1, 2, 0], 'int32');
    const res = tf.oneHot(indices, 3);

    expect(res.shape).toEqual([4, 3]);
    expectArraysClose(await res.data(), [0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0]);
  });

  it('Out of range events do not trigger onValue', async () => {
    const indices = tf.tensor1d([-1, 5, 12345], 'int32');
    const res = tf.oneHot(indices, 5);
    expect(res.shape).toEqual([3, 5]);
    expectArraysClose(
        await res.data(), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('Depth 2 onValue=3, offValue=-2', async () => {
    const indices = tf.tensor1d([0, 1], 'int32');
    const res = tf.oneHot(indices, 2, 3, -2);

    expect(res.shape).toEqual([2, 2]);
    expectArraysClose(await res.data(), [3, -2, -2, 3]);
  });

  it('indices not int32 throws error', () => {
    const indices = tf.tensor1d([0, 1], 'float32');
    expect(() => tf.oneHot(indices, 2)).toThrowError();
  });

  it('check output dtype', () => {
    const expectedType = 'int32';
    const indices = tf.tensor1d([0, 1], 'int32');
    const res = tf.oneHot(indices, 2);

    expect(res.dtype).toEqual(expectedType);
  });

  it('oneHot accepts a tensor-like object', async () => {
    const res = tf.oneHot([0, 1], 2);
    expect(res.shape).toEqual([2, 2]);
    expectArraysClose(await res.data(), [1, 0, 0, 1]);
  });

  it('has gradient', async () => {
    const a = tf.tensor1d([0, 1, 2], 'int32');
    const dy = tf.ones([3, 3], 'float32');
    const da = tf.grad((x: tf.Tensor1D) => tf.oneHot(x, 3))(a, dy);

    expect(da.dtype).toBe('float32');
    expect(da.shape).toEqual([3]);
    expectArraysClose(await da.data(), [0, 0, 0]);
  });

  it('gradient with clones', async () => {
    const a = tf.tensor1d([0, 1, 2], 'int32');
    const dy = tf.ones([3, 3], 'float32');
    const da =
        tf.grad((x: tf.Tensor1D) => tf.oneHot(x.clone(), 3).clone())(a, dy);

    expect(da.dtype).toBe('float32');
    expect(da.shape).toEqual([3]);
    expectArraysClose(await da.data(), [0, 0, 0]);
  });

  it('gradient when indices is 3d', async () => {
    const a = tf.tensor3d([1, 2, 3, 4], [1, 2, 2], 'int32');
    const dy = tf.ones([1, 2, 2, 3], 'float32');
    const depth = 3;
    const da = tf.grad(x => tf.oneHot(x, depth))(a, dy);
    expect(da.dtype).toBe('float32');
    expect(da.shape).toEqual(a.shape);
    expectArraysClose(await da.data(), [0, 0, 0, 0]);
  });

  it('oneHot with indices as 2d', async () => {
    const indices = tf.tensor2d([[1, 3], [2, 3]], [2, 2], 'int32');
    const depth = 4;
    const res = tf.oneHot(indices, depth);
    expect(res.shape).toEqual([2, 2, depth]);
    expectArraysClose(
        await res.data(), [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]);
  });

  it('Supports chaining', async () => {
    const indices =
        tf.tensor2d([[1, 2, 3], [2, 3, 1], [4, 5, 6]], [3, 3], 'int32');
    const depth = 6;
    const onValue = 3;
    const offValue = 7;
    const res = indices.oneHot(depth, onValue, offValue);

    expect(res.shape).toEqual([3, 3, 6]);
    expectArraysClose(await res.data(), [
      7, 3, 7, 7, 7, 7, 7, 7, 3, 7, 7, 7, 7, 7, 7, 3, 7, 7,
      7, 7, 3, 7, 7, 7, 7, 7, 7, 3, 7, 7, 7, 3, 7, 7, 7, 7,
      7, 7, 7, 7, 3, 7, 7, 7, 7, 7, 7, 3, 7, 7, 7, 7, 7, 7
    ]);
  });
});
