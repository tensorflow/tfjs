/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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

describeWithFlags('expm1', ALL_ENVS, () => {
  it('expm1', async () => {
    const a = tf.tensor1d([1, 2, 0]);
    const r = tf.expm1(a);

    expectArraysClose(
        await r.data(), [Math.expm1(1), Math.expm1(2), Math.expm1(0)]);
  });

  it('expm1 propagates NaNs', async () => {
    const a = tf.tensor1d([1, NaN, 0]);
    const r = tf.expm1(a);
    expectArraysClose(await r.data(), [Math.expm1(1), NaN, Math.expm1(0)]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(0.5);
    const dy = tf.scalar(3);

    const gradients = tf.grad(a => tf.expm1(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [3 * Math.exp(0.5)]);
  });

  it('gradient with clones', async () => {
    const a = tf.scalar(0.5);
    const dy = tf.scalar(3);

    const gradients = tf.grad(a => tf.expm1(a.clone()).clone())(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [3 * Math.exp(0.5)]);
  });

  it('gradients: Tensor1D', async () => {
    const a = tf.tensor1d([-1, 2, 3, -5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.expm1(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        await gradients.data(),
        [1 * Math.exp(-1), 2 * Math.exp(2), 3 * Math.exp(3), 4 * Math.exp(-5)],
        1e-1);
  });

  it('gradients: Tensor2D', async () => {
    const a = tf.tensor2d([-3, 1, 2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.expm1(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        await gradients.data(),
        [1 * Math.exp(-3), 2 * Math.exp(1), 3 * Math.exp(2), 4 * Math.exp(3)],
        1e-1);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.expm1({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'expm1' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const r = tf.expm1([1, 2, 0]);

    expectArraysClose(
        await r.data(), [Math.expm1(1), Math.expm1(2), Math.expm1(0)]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.expm1('q'))
        .toThrowError(/Argument 'x' passed to 'expm1' must be numeric/);
  });
});
