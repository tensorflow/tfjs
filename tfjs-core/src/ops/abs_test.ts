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

describeWithFlags('abs', ALL_ENVS, () => {
  it('basic', async () => {
    const a = tf.tensor1d([1, -2, 0, 3, -0.1]);
    const result = tf.abs(a);
    expectArraysClose(await result.data(), [1, 2, 0, 3, 0.1]);
  });

  it('5D', async () => {
    const a = tf.tensor5d([1, -2, 0, -3], [1, 2, 2, 1, 1]);
    const result = tf.abs(a);
    expectArraysClose(await result.data(), [1, 2, 0, 3]);
  });

  it('6D', async () => {
    const a = tf.tensor6d([1, -2, 5, -3, -1, 4, 7, 8], [1, 2, 2, 2, 1, 1]);
    const result = tf.abs(a);
    expectArraysClose(await result.data(), [1, 2, 5, 3, 1, 4, 7, 8]);
  });

  it('complex64 rank-1', async () => {
    const a = tf.complex([-2, -1, 0, 1, 2], [1, 2, 3, 0, -1]);
    const result = tf.abs(a);
    expectArraysClose(await result.data(), [
      Math.sqrt(-2 * -2 + 1 * 1), Math.sqrt(-1 * -1 + 2 * 2),
      Math.sqrt(0 * 0 + 3 * 3), Math.sqrt(1 * 1 + 0 * 0),
      Math.sqrt(2 * 2 + -1 * -1)
    ]);
    expect(result.shape).toEqual([5]);
  });

  it('complex64 rank-2', async () => {
    const a = tf.complex([[-3, -2, -1], [0, 1, 2]], [[4, 1, 2], [3, 0, -1]]);
    const result = tf.abs(a);
    expectArraysClose(await result.data(), [
      Math.sqrt(-3 * -3 + 4 * 4), Math.sqrt(-2 * -2 + 1 * 1),
      Math.sqrt(-1 * -1 + 2 * 2), Math.sqrt(0 * 0 + 3 * 3),
      Math.sqrt(1 * 1 + 0 * 0), Math.sqrt(2 * 2 + -1 * -1)
    ]);
    expect(result.shape).toEqual([2, 3]);
  });

  it('complex64 rank-3', async () => {
    const a = tf.complex(
        [[[-3, -2], [-1, 0]], [[1, 2], [3, 4]]],
        [[[4, 1], [2, 3]], [[0, -1], [-3, -4]]]);
    const result = tf.abs(a);
    expectArraysClose(await result.data(), [
      Math.sqrt(-3 * -3 + 4 * 4), Math.sqrt(-2 * -2 + 1 * 1),
      Math.sqrt(-1 * -1 + 2 * 2), Math.sqrt(0 * 0 + 3 * 3),
      Math.sqrt(1 * 1 + 0 * 0), Math.sqrt(2 * 2 + -1 * -1),
      Math.sqrt(3 * 3 + -3 * -3), Math.sqrt(4 * 4 + -4 * -4)
    ]);
    expect(result.shape).toEqual([2, 2, 2]);
  });

  it('is underflow-safe for complex64', async () => {
    const floatBits = tf.backend().floatPrecision();
    let small;
    switch (floatBits) {
      case 32:
        small = 1e-30;
        break;
      case 16:
        small = 1e-4;
        break;
      default:
        throw new Error(`Test not implemented for ENV.engine.floatPrecision()=${
            floatBits}.`);
    }

    const a = tf.complex([small, 0, small, 0], [small, small, 0, 0]);
    const result = tf.abs(a);
    expectArraysClose(
        await result.data(),
        [
          Math.hypot(small, small), Math.hypot(0, small), Math.hypot(small, 0),
          Math.hypot(0, 0)
        ],
        /*tolerance=*/small / 100);
    expect(result.shape).toEqual([4]);
  });

  it('propagates NaNs', async () => {
    const a = tf.tensor1d([1, -2, 0, 3, -0.1, NaN]);
    const result = tf.abs(a);
    expectArraysClose(await result.data(), [1, 2, 0, 3, 0.1, NaN]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(4);
    const dy = tf.scalar(8);

    const da = tf.grad(a => tf.abs(a))(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [8 * 1]);
  });

  it('gradient with clones', () => {
    const a = tf.scalar(4);
    const dy = tf.scalar(8);

    const da = tf.grad(a => a.clone().abs().clone())(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
  });

  it('gradients: Tensor1D', async () => {
    const a = tf.tensor1d([1, 2, -3, 5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const da = tf.grad(a => tf.abs(a))(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [1 * 1, 2 * 1, 3 * -1, 4 * 1]);
  });

  it('gradients: Tensor2D', async () => {
    const a = tf.tensor2d([3, -1, -2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const da = tf.grad(a => tf.abs(a))(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [1 * 1, 2 * -1, 3 * -1, 4 * 1]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.abs({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'abs' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.abs([1, -2, 0, 3, -0.1]);
    expectArraysClose(await result.data(), [1, 2, 0, 3, 0.1]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.abs('q'))
        .toThrowError(/Argument 'x' passed to 'abs' must be numeric/);
  });
});
