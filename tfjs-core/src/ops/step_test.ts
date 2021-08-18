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

describeWithFlags('step kernel', ALL_ENVS, () => {
  it('with 1d tensor', async () => {
    const a = tf.tensor1d([1, -2, -.01, 3, -0.1]);
    const result = tf.step(a);
    expectArraysClose(await result.data(), [1, 0, 0, 1, 0]);
  });

  it('with 1d tensor and alpha', async () => {
    const a = tf.tensor1d([1, -2, -.01, 3, NaN]);
    const result = tf.step(a, 0.1);
    expectArraysClose(await result.data(), [1, 0.1, 0.1, 1, NaN]);
  });

  it('with 2d tensor', async () => {
    const a = tf.tensor2d([1, -5, -3, 4], [2, 2]);
    const result = tf.step(a);
    expect(result.shape).toEqual([2, 2]);
    expectArraysClose(await result.data(), [1, 0, 0, 1]);
  });

  it('propagates NaNs', async () => {
    const a = tf.tensor1d([1, -2, -.01, 3, NaN]);
    const result = tf.step(a);
    expectArraysClose(await result.data(), [1, 0, 0, 1, NaN]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(-4);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.step(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0]);
  });

  it('gradient with clones', async () => {
    const a = tf.scalar(-4);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.step(a.clone()).clone())(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0]);
  });

  it('gradients: Tensor1D', async () => {
    const a = tf.tensor1d([1, 2, -3, 5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.step(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0, 0, 0, 0]);
  });

  it('gradients: Tensor2D', async () => {
    const a = tf.tensor2d([3, -1, -2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.step(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [0, 0, 0, 0]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.step({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'step' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.step([1, -2, -.01, 3, -0.1]);
    expectArraysClose(await result.data(), [1, 0, 0, 1, 0]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.step('q'))
        .toThrowError(/Argument 'x' passed to 'step' must be numeric/);
  });
});
