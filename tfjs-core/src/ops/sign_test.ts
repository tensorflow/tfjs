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

describeWithFlags('sign', ALL_ENVS, () => {
  it('basic', async () => {
    const a = tf.tensor1d([1.5, 0, NaN, -1.4]);
    const r = tf.sign(a);
    expectArraysClose(await r.data(), [1, 0, 0, -1]);
  });

  it('does not propagate NaNs', async () => {
    const a = tf.tensor1d([1.5, NaN, -1.4]);
    const r = tf.sign(a);
    expectArraysClose(await r.data(), [1, 0, -1]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(5.2);
    const dy = tf.scalar(3);

    const gradients = tf.grad(a => tf.sign(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0]);
  });

  it('gradient with clones', async () => {
    const a = tf.scalar(5.2);
    const dy = tf.scalar(3);

    const gradients = tf.grad(a => tf.sign(a.clone()).clone())(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0]);
  });

  it('gradients: Tensor1D', async () => {
    const a = tf.tensor1d([-1.1, 2.6, 3, -5.9]);
    const dy = tf.tensor1d([-1, 1, 1, -1]);

    const gradients = tf.grad(a => tf.sign(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0, 0, 0, 0]);
  });

  it('gradients: Tensor2D', async () => {
    const a = tf.tensor2d([-3, 1, 2.2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.sign(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0, 0, 0, 0]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.sign({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'sign' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const r = tf.sign([1.5, 0, NaN, -1.4]);
    expectArraysClose(await r.data(), [1, 0, 0, -1]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.sign('q'))
        .toThrowError(/Argument 'x' passed to 'sign' must be numeric/);
  });
});
