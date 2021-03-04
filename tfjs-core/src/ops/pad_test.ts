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

describeWithFlags('pad 1d', ALL_ENVS, () => {
  it('Should pad 1D arrays', async () => {
    const a = tf.tensor1d([1, 2, 3, 4, 5, 6], 'int32');
    const b = tf.pad1d(a, [2, 3]);
    expectArraysClose(await b.data(), [0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0]);
  });

  it('Should not pad 1D arrays with 0s', async () => {
    const a = tf.tensor1d([1, 2, 3, 4], 'int32');
    const b = tf.pad1d(a, [0, 0]);
    expectArraysClose(await b.data(), [1, 2, 3, 4]);
  });

  it('Should handle padding with custom value', async () => {
    let a = tf.tensor1d([1, 2, 3, 4], 'int32');
    let b = tf.pad1d(a, [2, 3], 9);
    expectArraysClose(await b.data(), [9, 9, 1, 2, 3, 4, 9, 9, 9]);

    a = tf.tensor1d([1, 2, 3, 4]);
    b = tf.pad1d(a, [2, 1], 1.1);
    expectArraysClose(await b.data(), [1.1, 1.1, 1, 2, 3, 4, 1.1]);

    a = tf.tensor1d([1, 2, 3, 4]);
    b = tf.pad1d(a, [2, 1], 1);
    expectArraysClose(await b.data(), [1, 1, 1, 2, 3, 4, 1]);

    a = tf.tensor1d([1, 2, 3, 4]);
    b = tf.pad1d(a, [2, 1], Number.NEGATIVE_INFINITY);
    expectArraysClose(await b.data(), [
      Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, 1, 2, 3, 4,
      Number.NEGATIVE_INFINITY
    ]);

    a = tf.tensor1d([1, 2, 3, 4]);
    b = tf.pad1d(a, [2, 1], Number.POSITIVE_INFINITY);
    expectArraysClose(await b.data(), [
      Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, 1, 2, 3, 4,
      Number.POSITIVE_INFINITY
    ]);
  });

  it('Should handle NaNs with 1D arrays', async () => {
    const a = tf.tensor1d([1, NaN, 2, NaN]);
    const b = tf.pad1d(a, [1, 1]);
    expectArraysClose(await b.data(), [0, 1, NaN, 2, NaN, 0]);
  });

  it('Should handle invalid paddings', () => {
    const a = tf.tensor1d([1, 2, 3, 4], 'int32');
    const f = () => {
      // tslint:disable-next-line:no-any
      tf.pad1d(a, [2, 2, 2] as any);
    };
    expect(f).toThrowError();
  });

  it('grad', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const dy = tf.tensor1d([10, 20, 30, 40, 50, 60]);
    const da = tf.grad((a: tf.Tensor1D) => tf.pad1d(a, [2, 1]))(a, dy);
    expect(da.shape).toEqual([3]);
    expectArraysClose(await da.data(), [30, 40, 50]);
  });

  it('gradient with clones', async () => {
    const a = tf.tensor1d([1, 2, 3]);
    const dy = tf.tensor1d([10, 20, 30, 40, 50, 60]);
    const da =
        tf.grad((a: tf.Tensor1D) => tf.pad1d(a.clone(), [2, 1]).clone())(a, dy);
    expect(da.shape).toEqual([3]);
    expectArraysClose(await da.data(), [30, 40, 50]);
  });

  it('accepts a tensor-like object', async () => {
    const a = [1, 2, 3, 4, 5, 6];
    const b = tf.pad1d(a, [2, 3]);
    expectArraysClose(await b.data(), [0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0]);
  });
});

describeWithFlags('pad 2d', ALL_ENVS, () => {
  it('Should pad 2D arrays', async () => {
    let a = tf.tensor2d([[1], [2]], [2, 1], 'int32');
    let b = tf.pad2d(a, [[1, 1], [1, 1]]);
    // 0, 0, 0
    // 0, 1, 0
    // 0, 2, 0
    // 0, 0, 0
    expectArraysClose(await b.data(), [0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0]);

    a = tf.tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3], 'int32');
    b = tf.pad2d(a, [[2, 2], [1, 1]]);
    // 0, 0, 0, 0, 0
    // 0, 0, 0, 0, 0
    // 0, 1, 2, 3, 0
    // 0, 4, 5, 6, 0
    // 0, 0, 0, 0, 0
    // 0, 0, 0, 0, 0
    expectArraysClose(await b.data(), [
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0,
      0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]);
  });

  it('Should not pad 2D arrays with 0s', async () => {
    const a = tf.tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3], 'int32');
    const b = tf.pad2d(a, [[0, 0], [0, 0]]);
    expectArraysClose(await b.data(), [1, 2, 3, 4, 5, 6]);
  });

  it('Should handle padding with custom value', async () => {
    let a = tf.tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3], 'int32');
    let b = tf.pad2d(a, [[1, 1], [1, 1]], 10);
    expectArraysClose(await b.data(), [
      10, 10, 10, 10, 10, 10, 1,  2,  3,  10,
      10, 4,  5,  6,  10, 10, 10, 10, 10, 10
    ]);

    a = tf.tensor2d([[1], [1]], [2, 1]);
    b = tf.pad2d(a, [[1, 1], [1, 1]], -2.1);
    expectArraysClose(
        await b.data(),
        [-2.1, -2.1, -2.1, -2.1, 1, -2.1, -2.1, 1, -2.1, -2.1, -2.1, -2.1]);

    a = tf.tensor2d([[1], [1]], [2, 1]);
    b = tf.pad2d(a, [[1, 1], [1, 1]], -2);
    expectArraysClose(
        await b.data(), [-2, -2, -2, -2, 1, -2, -2, 1, -2, -2, -2, -2]);
  });

  it('Should handle NaNs with 2D arrays', async () => {
    const a = tf.tensor2d([[1, NaN], [1, NaN]], [2, 2]);
    const b = tf.pad2d(a, [[1, 1], [1, 1]]);
    // 0, 0, 0,   0
    // 0, 1, NaN, 0
    // 0, 1, NaN, 0
    // 0, 0, 0,   0
    expectArraysClose(
        await b.data(), [0, 0, 0, 0, 0, 1, NaN, 0, 0, 1, NaN, 0, 0, 0, 0, 0]);
  });

  it('Should handle invalid paddings', () => {
    const a = tf.tensor2d([[1], [2]], [2, 1], 'int32');
    const f = () => {
      // tslint:disable-next-line:no-any
      tf.pad2d(a, [[2, 2, 2], [1, 1, 1]] as any);
    };
    expect(f).toThrowError();
  });

  it('grad', async () => {
    const a = tf.tensor2d([[1, 2], [3, 4]]);
    const dy = tf.tensor2d([[0, 0, 0], [10, 20, 0], [30, 40, 0]], [3, 3]);
    const da =
        tf.grad((a: tf.Tensor2D) => tf.pad2d(a, [[1, 0], [0, 1]]))(a, dy);
    expect(da.shape).toEqual([2, 2]);
    expectArraysClose(await da.data(), [10, 20, 30, 40]);
  });

  it('accepts a tensor-like object', async () => {
    const a = [[1, 2, 3], [4, 5, 6]];  // 2x3
    const b = tf.pad2d(a, [[0, 0], [0, 0]]);
    expectArraysClose(await b.data(), [1, 2, 3, 4, 5, 6]);
  });
});

describeWithFlags('pad 3d', ALL_ENVS, () => {
  it('works with 3d tensor, float32', async () => {
    const a = tf.tensor3d([[[1]], [[2]]], [2, 1, 1], 'float32');
    const b = tf.pad3d(a, [[1, 1], [1, 1], [1, 1]]);
    // 0, 0, 0
    // 0, 0, 0
    // 0, 0, 0

    // 0, 0, 0
    // 0, 1, 0
    // 0, 0, 0

    // 0, 0, 0
    // 0, 2, 0
    // 0, 0, 0

    // 0, 0, 0
    // 0, 0, 0
    // 0, 0, 0
    expect(b.shape).toEqual([4, 3, 3]);
    expectArraysClose(await b.data(), [
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]);
  });
});

describeWithFlags('pad 4d', ALL_ENVS, () => {
  it('Should pad 4D arrays', async () => {
    const a = tf.tensor4d([[[[9]]]], [1, 1, 1, 1], 'int32');
    const b = tf.pad4d(a, [[0, 0], [1, 1], [1, 1], [0, 0]]);
    const expected = tf.tensor4d(
        [[[[0], [0], [0]], [[0], [9], [0]], [[0], [0], [0]]]], [1, 3, 3, 1],
        'int32');
    expectArraysClose(await b.data(), await expected.data());
    expect(b.dtype).toBe('int32');
    expect(b.shape).toEqual([1, 3, 3, 1]);
  });

  it('does not leak memory', () => {
    const a = tf.tensor4d([[[[9]]]], [1, 1, 1, 1], 'int32');
    // The first call to pad may create and keeps internal singleton tensors.
    // Subsequent calls should always create exactly one new tensor.
    tf.pad4d(a, [[0, 0], [1, 1], [1, 1], [0, 0]]);
    // Count before real call.
    const numTensors = tf.memory().numTensors;
    tf.pad4d(a, [[0, 0], [1, 1], [1, 1], [0, 0]]);
    expect(tf.memory().numTensors).toEqual(numTensors + 1);
  });

  it('accepts a tensor-like object', async () => {
    const a = [[[[9]]]];  // 1x1x1x1
    const b = tf.pad4d(a, [[0, 0], [1, 1], [1, 1], [0, 0]]);
    const expected = tf.tensor4d(
        [[[[0], [0], [0]], [[0], [9], [0]], [[0], [0], [0]]]], [1, 3, 3, 1],
        'float32');
    expectArraysClose(await b.data(), await expected.data());
    expect(b.dtype).toBe('float32');
    expect(b.shape).toEqual([1, 3, 3, 1]);
  });
});

describeWithFlags('pad', ALL_ENVS, () => {
  it('Pad tensor2d', async () => {
    let a = tf.tensor2d([[1], [2]], [2, 1], 'int32');
    let b = tf.pad(a, [[1, 1], [1, 1]]);
    // 0, 0, 0
    // 0, 1, 0
    // 0, 2, 0
    // 0, 0, 0
    expectArraysClose(await b.data(), [0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0]);

    a = tf.tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3], 'int32');
    b = tf.pad(a, [[2, 2], [1, 1]]);
    // 0, 0, 0, 0, 0
    // 0, 0, 0, 0, 0
    // 0, 1, 2, 3, 0
    // 0, 4, 5, 6, 0
    // 0, 0, 0, 0, 0
    // 0, 0, 0, 0, 0
    expectArraysClose(await b.data(), [
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0,
      0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.pad({} as tf.Tensor, [[0, 0]]))
        .toThrowError(/Argument 'x' passed to 'pad' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const x = [[1], [2]];
    const res = tf.pad(x, [[1, 1], [1, 1]]);
    // 0, 0, 0
    // 0, 1, 0
    // 0, 2, 0
    // 0, 0, 0
    expectArraysClose(await res.data(), [0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0]);
  });
});
