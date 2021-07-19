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
import {expectArraysClose, TEST_EPSILON_FLOAT16} from '../test_util';

describeWithFlags('tan', ALL_ENVS, () => {
  it('basic', async () => {
    const values = [1, -3, 2, 7, -4];
    const a = tf.tensor1d(values);
    const result = tf.tan(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.tan(values[i]);
    }
    expectArraysClose(await result.data(), expected);
  });

  it('propagates NaNs', async () => {
    const a = tf.tensor1d([4, NaN, 0]);
    const res = tf.tan(a);
    expectArraysClose(await res.data(), [Math.tan(4), NaN, Math.tan(0)]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(0.5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.tan(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        await gradients.data(), [8 / (Math.cos(0.5) * Math.cos(0.5))]);
  });

  it('gradient with clones', async () => {
    const a = tf.scalar(0.5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.tan(a.clone()).clone())(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        await gradients.data(), [8 / (Math.cos(0.5) * Math.cos(0.5))]);
  });

  it('gradients: Tensor1D', async () => {
    const aValues = [-1, 2, 3, -5];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor1d(aValues);
    const dy = tf.tensor1d(dyValues);

    const gradients = tf.grad(a => tf.tan(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] / (Math.cos(aValues[i]) * Math.cos(aValues[i]));
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    // The grad(tan(x)) which relies on 1/cos(x) is less precise on Windows.
    expectArraysClose(await gradients.data(), expected, TEST_EPSILON_FLOAT16);
  });

  it('gradients: Tensor2D', async () => {
    const aValues = [-3, 1, 2, 3];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor2d(aValues, [2, 2]);
    const dy = tf.tensor2d(dyValues, [2, 2]);

    const gradients = tf.grad(a => tf.tan(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] / (Math.cos(aValues[i]) * Math.cos(aValues[i]));
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), expected);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.tan({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'tan' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const values = [1, -3, 2, 7, -4];
    const result = tf.tan(values);

    const expected = [];
    for (let i = 0; i < values.length; i++) {
      expected[i] = Math.tan(values[i]);
    }
    expectArraysClose(await result.data(), expected);
  });

  it('throws for string tensor', () => {
    expect(() => tf.tan('q'))
        .toThrowError(/Argument 'x' passed to 'tan' must be numeric/);
  });
});
