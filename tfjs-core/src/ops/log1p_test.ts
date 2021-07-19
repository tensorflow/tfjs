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

describeWithFlags('log1p', ALL_ENVS, () => {
  it('log1p', async () => {
    const a = tf.tensor1d([1, 2]);
    const r = tf.log1p(a);
    expectArraysClose(await r.data(), [Math.log1p(1), Math.log1p(2)]);
  });

  it('log1p propagates NaNs', async () => {
    const a = tf.tensor1d([1, NaN]);
    const r = tf.log1p(a);
    expectArraysClose(await r.data(), [Math.log1p(1), NaN]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(5);
    const dy = tf.scalar(3);

    const gradients = tf.grad(a => tf.log1p(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [3 / (1 + 5)]);
  });

  it('gradient with clones', () => {
    const a = tf.scalar(5);
    const gradients = tf.grad(a => a.clone().log1p().clone())(a);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
  });

  it('gradients: Tensor1D', async () => {
    const a = tf.tensor1d([-1, 2, 3, -5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.log1p(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        await gradients.data(),
        [Infinity, 2 / (1 + 2), 3 / (1 + 3), 4 / (1 + -5)]);
  });

  it('gradients: Tensor2D', async () => {
    const a = tf.tensor2d([-3, 1, 2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.log1p(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        await gradients.data(),
        [1 / (1 + -3), 2 / (1 + 1), 3 / (1 + 2), 4 / (1 + 3)]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.log1p({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'log1p' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const r = tf.log1p([1, 2]);
    expectArraysClose(await r.data(), [Math.log1p(1), Math.log1p(2)]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.log1p('q'))
        .toThrowError(/Argument 'x' passed to 'log1p' must be numeric/);
  });
});
