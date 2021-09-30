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

describeWithFlags('rsqrt', ALL_ENVS, () => {
  it('rsqrt', async () => {
    const a = tf.tensor1d([2, 4]);
    const r = tf.rsqrt(a);
    expectArraysClose(await r.data(), [1 / Math.sqrt(2), 1 / Math.sqrt(4)]);
  });

  it('rsqrt propagates NaNs', async () => {
    const a = tf.tensor1d([1, NaN]);
    const r = tf.rsqrt(a);
    expectArraysClose(await r.data(), [1 / Math.sqrt(1), NaN]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(4);
    const dy = tf.scalar(8);

    const da = tf.grad(a => tf.rsqrt(a))(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [(-1 * 8) / (2 * Math.pow(4, 1.5))]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(4);
    const dy = tf.scalar(8);

    const da = tf.grad(a => tf.rsqrt(a.clone()).clone())(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [(-1 * 8) / (2 * Math.pow(4, 1.5))]);
  });

  it('gradients: Tensor1D', async () => {
    const a = tf.tensor1d([1, 2, 3, 5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.rsqrt(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        await gradients.data(),
        [
          -1 * 1 / (2 * Math.pow(1, 1.5)), -1 * 2 / (2 * Math.pow(2, 1.5)),
          -1 * 3 / (2 * Math.pow(3, 1.5)), -1 * 4 / (2 * Math.pow(5, 1.5))
        ],
        1e-1);
  });

  it('gradients: Tensor2D', async () => {
    const a = tf.tensor2d([3, 1, 2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.rsqrt(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        await gradients.data(),
        [
          -1 * 1 / (2 * Math.pow(3, 1.5)), -1 * 2 / (2 * Math.pow(1, 1.5)),
          -1 * 3 / (2 * Math.pow(2, 1.5)), -1 * 4 / (2 * Math.pow(3, 1.5))
        ],
        1e-1);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.rsqrt({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'rsqrt' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const r = tf.rsqrt([2, 4]);
    expectArraysClose(await r.data(), [1 / Math.sqrt(2), 1 / Math.sqrt(4)]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.rsqrt('q'))
        .toThrowError(/Argument 'x' passed to 'rsqrt' must be float32/);
  });

  it('throws for int32 tensor', () => {
    expect(() => tf.rsqrt(tf.tensor1d([1], 'int32')))
        .toThrowError(/Argument 'x' passed to 'rsqrt' must be float32/);
  });
});
