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

describeWithFlags('asin', ALL_ENVS, () => {
  it('basic', async () => {
    const values = [.1, -3, 2, 7, -4];
    const a = tf.tensor1d(values);
    const result = tf.asin(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.asin(values[i]);
    }
    expectArraysClose(await result.data(), expected);
  });

  it('propagates NaNs', async () => {
    const a = tf.tensor1d([4, NaN, 0]);
    const res = tf.asin(a);
    expectArraysClose(await res.data(), [Math.asin(4), NaN, Math.asin(0)]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(0.5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.asin(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [8 / Math.sqrt(1 - (0.5 * 0.5))]);
  });

  it('gradient with clones', async () => {
    const a = tf.scalar(0.5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.asin(a.clone()).clone())(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [8 / Math.sqrt(1 - (0.5 * 0.5))]);
  });

  it('gradients: Tensor1D', async () => {
    const aValues = [-0.1, 0.2, 0.3, -0.5];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor1d(aValues);
    const dy = tf.tensor1d(dyValues);

    const gradients = tf.grad(a => tf.asin(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] / Math.sqrt(1 - (aValues[i] * aValues[i]));
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), expected);
  });

  it('gradients: Tensor2D', async () => {
    const aValues = [-0.3, 0.1, 0.2, 0.3];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor2d(aValues, [2, 2]);
    const dy = tf.tensor2d(dyValues, [2, 2]);

    const gradients = tf.grad(a => tf.asin(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] / Math.sqrt(1 - (aValues[i] * aValues[i]));
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), expected);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.asin({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'asin' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const values = [.1, -3, 2, 7, -4];
    const result = tf.asin(values);

    const expected = [];
    for (let i = 0; i < values.length; i++) {
      expected[i] = Math.asin(values[i]);
    }
    expectArraysClose(await result.data(), expected);
  });

  it('throws for string tensor', () => {
    expect(() => tf.asin('q'))
        .toThrowError(/Argument 'x' passed to 'asin' must be numeric/);
  });
});
