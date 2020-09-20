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

describeWithFlags('mirrorPad', ALL_ENVS, () => {
  it('MirrorPad tensor1d', async () => {
    const a = tf.tensor1d([1, 2, 3], 'int32');
    let b = tf.mirrorPad(a, [[2, 2]]);
    expectArraysClose(await b.data(), [3, 2, 1, 2, 3, 2, 1]);

    b = tf.mirrorPad(a, [[2, 2]], 'symmetric');
    expectArraysClose(await b.data(), [2, 1, 1, 2, 3, 3, 2]);
  });

  it('MirrorPad tensor2d', async () => {
    const a = tf.tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3], 'int32');
    let b = tf.mirrorPad(a, [[1, 1], [1, 1]]);
    // 5, 4, 5, 6, 5
    // 2, 1, 2, 3, 2
    // 5, 4, 5, 6, 5
    // 2, 1, 2, 3, 2
    expectArraysClose(
        await b.data(),
        [5, 4, 5, 6, 5, 2, 1, 2, 3, 2, 5, 4, 5, 6, 5, 2, 1, 2, 3, 2]);

    b = tf.mirrorPad(a, [[1, 1], [1, 1]], 'symmetric');
    // 1, 1, 2, 3, 3
    // 1, 1, 2, 3, 3
    // 4, 4, 5, 6, 6
    // 4, 4, 5, 6, 6
    expectArraysClose(
        await b.data(),
        [1, 1, 2, 3, 3, 1, 1, 2, 3, 3, 4, 4, 5, 6, 6, 4, 4, 5, 6, 6]);
  });

  it('MirrorPad tensor3d', async () => {
    const a = tf.tensor3d([[[1, 2]], [[3, 4]]], [2, 1, 2], 'int32');
    let b = tf.mirrorPad(a, [[1, 1], [0, 0], [1, 1]]);
    // 4, 3, 4, 3

    // 2, 1, 2, 1

    // 4, 3, 4, 3

    // 2, 1, 2, 1
    expect(b.shape).toEqual([4, 1, 4]);
    expectArraysClose(
        await b.data(), [4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2,
        1]);

    b = tf.mirrorPad(a, [[1, 1], [0, 0], [1, 1]], 'symmetric');
    // 1, 1, 2, 2

    // 1, 1, 2, 2

    // 3, 3, 4, 4

    // 3, 3, 4, 4
    expectArraysClose(
        await b.data(), [1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4,
        4]);
  });

  it('MirrorPad tensor4d', async () => {
    const a = tf.tensor4d([[[[1, 2, 3, 4]]]], [1, 1, 1, 4], 'int32');
    let b = tf.mirrorPad(a, [[0, 0], [0, 0], [0, 0], [1, 1]]);
    let expected = tf.tensor4d([[[[2, 1, 2, 3, 4, 3]]]], [1, 1, 1, 6],
    'int32'); expectArraysClose(await b.data(), await expected.data());
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([1, 1, 1, 6]);

    b = tf.mirrorPad(a, [[0, 0], [0, 0], [0, 0], [1, 1]], 'symmetric');
    expected = tf.tensor4d([[[[1, 1, 2, 3, 4, 4]]]], [1, 1, 1, 6],
    'int32');
    expectArraysClose(await b.data(), await expected.data());
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.mirrorPad({} as tf.Tensor, [[0, 0]]))
        .toThrowError(/Argument 'x' passed to 'mirrorPad' must be a Tensor/);
  });

  it('does not leak memory', () => {
    const a = tf.tensor4d([[[[1, 2, 3, 4]]]], [1, 1, 1, 4], 'int32');
    // The first call to mirrorPad may create and keeps internal
    // singleton tensors. Subsequent calls should always create exactly
    // one new tensor.
    tf.mirrorPad(a, [[0, 0], [0, 0], [0, 0], [1, 1]]);
    // Count before real call.
    const numTensors = tf.memory().numTensors;
    tf.mirrorPad(a, [[0, 0], [0, 0], [0, 0], [1, 1]]);
    expect(tf.memory().numTensors).toEqual(numTensors + 1);
  });

  it('accepts a tensor-like object', async () => {
    const x = [[1, 2, 3], [4, 5, 6]];
    const res = tf.mirrorPad(x, [[1, 1], [1, 1]]);
    // 5, 4, 5, 6, 5
    // 2, 1, 2, 3, 2
    // 5, 4, 5, 6, 5
    // 2, 1, 2, 3, 2
    expectArraysClose(
        await res.data(),
        [5, 4, 5, 6, 5, 2, 1, 2, 3, 2, 5, 4, 5, 6, 5, 2, 1, 2, 3, 2]);
  });

  it('Should handle invalid paddings', () => {
    const a = tf.tensor1d([1, 2, 3, 4], 'int32');
    const f = () => {
      // tslint:disable-next-line:no-any
      tf.mirrorPad(a, [2, 2, 2] as any);
    };
    expect(f).toThrowError();
  });

  it('Should handle paddings that are out of range', () => {
    const a = tf.tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3], 'int32');
    const f = () => {
      // tslint:disable-next-line:no-any
      tf.mirrorPad(a, [[4, 1], [1, 1]]);
    };
    expect(f).toThrowError();
  });

  it('Should handle NaNs', async () => {
    const a = tf.tensor2d([[1, NaN], [1, NaN]], [2, 2]);
    const b = tf.mirrorPad(a, [[1, 1], [1, 1]]);
    // NaN, 1, NaN, 1
    // NaN, 1, NaN, 1
    // NaN, 1, NaN, 1
    // NaN, 1, NaN, 1
    expectArraysClose(
        await b.data(),
        [NaN, 1, NaN, 1, NaN, 1, NaN, 1, NaN, 1, NaN, 1, NaN, 1, NaN,
        1]);
  });

  it('grad', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const dy = tf.tensor1d([10, 20, 30, 40, 50, 60]);
    const da = tf.grad((a: tf.Tensor1D) => tf.mirrorPad(a, [[2, 1]]))(a, dy);
    expect(da.shape).toEqual([3]);
    expectArraysClose(await da.data(), [30, 40, 50]);
  });

  it('gradient with clones', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const dy = tf.tensor1d([10, 20, 30, 40, 50, 60]);
    const da =
        tf.grad((a: tf.Tensor1D) =>
        tf.mirrorPad(a.clone(), [[2, 1]]).clone())(a, dy);
    expect(da.shape).toEqual([3]);
    expectArraysClose(await da.data(), [30, 40, 50]);
  });
});
