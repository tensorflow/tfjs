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

describeWithFlags('leakyrelu', ALL_ENVS, () => {
  it('basic', async () => {
    const a = tf.tensor1d([0, 1, -2]);
    const result = tf.leakyRelu(a);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(await result.data(), [0, 1, -0.4]);
  });

  it('propagates NaN', async () => {
    const a = tf.tensor1d([0, 1, NaN]);
    const result = tf.leakyRelu(a);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(await result.data(), [0, 1, NaN]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(-4);
    const dy = tf.scalar(8);
    const alpha = 0.1;

    const gradients = tf.grad((a) => tf.leakyRelu(a, alpha))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [8 * alpha]);
  });

  it('gradient with clones', async () => {
    const a = tf.scalar(-4);
    const dy = tf.scalar(8);
    const alpha = 0.1;

    const gradients =
        tf.grad((a) => tf.leakyRelu(a.clone(), alpha).clone())(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [8 * alpha]);
  });

  it('gradients: Tensor1D', async () => {
    const aValues = [1, -1, 0.1];
    const dyValues = [1, 2, 3];
    const alpha = 0.1;

    const a = tf.tensor1d(aValues);
    const dy = tf.tensor1d(dyValues);

    const gradients = tf.grad((a) => tf.leakyRelu(a, alpha))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');

    expectArraysClose(await gradients.data(), [1, 2 * alpha, 3]);
  });

  it('gradients: Tensor2D', async () => {
    const aValues = [1, -1, 0.1, 0.5];
    const dyValues = [1, 2, 3, 4];
    const alpha = 0.1;

    const a = tf.tensor2d(aValues, [2, 2]);
    const dy = tf.tensor2d(dyValues, [2, 2]);

    const gradients = tf.grad((a) => tf.leakyRelu(a, alpha))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');

    expectArraysClose(await gradients.data(), [1, 2 * alpha, 3, 4]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.leakyRelu({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'leakyRelu' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.leakyRelu([0, 1, -2]);

    expect(result.shape).toEqual([3]);
    expectArraysClose(await result.data(), [0, 1, -0.4]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.leakyRelu('q'))
        .toThrowError(/Argument 'x' passed to 'leakyRelu' must be numeric/);
  });
});
