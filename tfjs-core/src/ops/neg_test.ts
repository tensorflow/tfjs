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

describeWithFlags('neg', ALL_ENVS, () => {
  it('basic', async () => {
    const a = tf.tensor1d([1, -3, 2, 7, -4]);
    const result = tf.neg(a);
    expectArraysClose(await result.data(), [-1, 3, -2, -7, 4]);
  });

  it('propagate NaNs', async () => {
    const a = tf.tensor1d([1, -3, 2, 7, NaN]);
    const result = tf.neg(a);
    const expected = [-1, 3, -2, -7, NaN];
    expectArraysClose(await result.data(), expected);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(4);
    const dy = tf.scalar(8);

    const da = tf.grad(a => tf.neg(a))(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [8 * -1]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(4);
    const dy = tf.scalar(8);

    const da = tf.grad(a => tf.neg(a.clone()).clone())(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [8 * -1]);
  });

  it('gradients: Tensor1D', async () => {
    const a = tf.tensor1d([1, 2, -3, 5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const da = tf.grad(a => tf.neg(a))(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [1 * -1, 2 * -1, 3 * -1, 4 * -1]);
  });

  it('gradients: Tensor2D', async () => {
    const a = tf.tensor2d([3, -1, -2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const da = tf.grad(a => tf.neg(a))(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [1 * -1, 2 * -1, 3 * -1, 4 * -1]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.neg({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'neg' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.neg([1, -3, 2, 7, -4]);
    expectArraysClose(await result.data(), [-1, 3, -2, -7, 4]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.neg('q'))
        .toThrowError(/Argument 'x' passed to 'neg' must be numeric/);
  });
});
