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

describeWithFlags('acosh', ALL_ENVS, () => {
  it('basic', async () => {
    const values = [2, 3, 4, 5, 6];
    const a = tf.tensor1d(values);
    const result = tf.acosh(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.acosh(values[i]);
    }
    expectArraysClose(await result.data(), expected);
  });

  it('scalar', async () => {
    const value = 2;
    const a = tf.scalar(value);
    const result = tf.acosh(a);

    const expected = [Math.acosh(value)];
    expectArraysClose(await result.data(), expected);
  });

  it('tensor2d', async () => {
    const values = [2, 3, 4, 5];
    const a = tf.tensor2d(values, [2, 2]);
    const result = tf.acosh(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.acosh(values[i]);
    }
    expectArraysClose(await result.data(), expected);
  });

  it('propagates NaNs', async () => {
    const a = tf.tensor1d([4, NaN, 2]);
    const res = tf.acosh(a);
    expectArraysClose(await res.data(), [Math.acosh(4), NaN, Math.acosh(2)]);
  });

  it('NaN outside function domain', async () => {
    const a = tf.tensor1d([4, -1, 2]);
    const res = tf.acosh(a);
    expectArraysClose(await res.data(), [Math.acosh(4), NaN, Math.acosh(2)]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(1.5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.acosh(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        await gradients.data(), [8.0 / Math.sqrt(1.5 * 1.5 - 1.0)]);
  });

  it('gradient with clones', async () => {
    const a = tf.scalar(1.5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.acosh(a.clone()).clone())(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        await gradients.data(), [8.0 / Math.sqrt(1.5 * 1.5 - 1.0)]);
  });

  it('gradients: Tensor1D', async () => {
    const aValues = [2, 3, 5, 10];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor1d(aValues);
    const dy = tf.tensor1d(dyValues);

    const gradients = tf.grad(a => tf.acosh(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] / Math.sqrt(Math.pow(aValues[i], 2) - 1.0);
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), expected);
  });

  it('gradients: Tensor2D', async () => {
    const aValues = [2, 3, 5, 7];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor2d(aValues, [2, 2]);
    const dy = tf.tensor2d(dyValues, [2, 2]);

    const gradients = tf.grad(a => tf.acosh(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] / Math.sqrt(Math.pow(aValues[i], 2) - 1.0);
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), expected);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.acosh({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'acosh' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const values = [2, 3, 4, 5, 6];
    const result = tf.acosh(values);

    const expected = [];
    for (let i = 0; i < values.length; i++) {
      expected[i] = Math.acosh(values[i]);
    }
    expectArraysClose(await result.data(), expected);
  });

  it('throws for string tensor', () => {
    expect(() => tf.acosh('q'))
        .toThrowError(/Argument 'x' passed to 'acosh' must be numeric/);
  });
});
