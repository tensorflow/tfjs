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

describeWithFlags('erf', ALL_ENVS, () => {
  it('basic', async () => {
    const values = [-0.25, 0.25, 0.5, .75, -0.4];
    const a = tf.tensor1d(values);
    const result = tf.erf(a);
    const expected = [-0.2763264, 0.2763264, 0.5204999, 0.7111556, -0.4283924];
    expectArraysClose(await result.data(), expected);
  });

  it('blowup', async () => {
    const values = [-1.4, -2.5, -3.1, -4.4];
    const a = tf.tensor1d(values);
    const result = tf.erf(a);
    const expected = [-0.9522852, -0.999593, -0.9999883, -1];
    expectArraysClose(await result.data(), expected);
  });

  it('scalar', async () => {
    const a = tf.scalar(1);
    const result = tf.erf(a);
    const expected = [0.8427008];
    expectArraysClose(await result.data(), expected);
  });

  it('scalar in int32', async () => {
    const a = tf.scalar(1, 'int32');
    const result = tf.erf(a);
    const expected = [0.8427008];
    expectArraysClose(await result.data(), expected);
  });

  it('tensor2d', async () => {
    const values = [0.2, 0.3, 0.4, 0.5];
    const a = tf.tensor2d(values, [2, 2]);
    const result = tf.erf(a);
    const expected = [0.2227026, 0.32862678, 0.42839235, 0.5204999];
    expectArraysClose(await result.data(), expected);
  });

  it('propagates NaNs', async () => {
    const a = tf.tensor1d([0.5, NaN, 0]);
    const res = tf.erf(a);
    expectArraysClose(await res.data(), [0.5204999, NaN, 0.0]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(0.5);
    const dy = tf.scalar(8);
    const gradients = tf.grad(a => tf.erf(a))(a, dy);
    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        await gradients.data(),
        [8 * 2 * Math.exp(-0.5 * 0.5) / Math.sqrt(Math.PI)]);
  });

  it('gradient with clones', async () => {
    const a = tf.scalar(0.5);
    const dy = tf.scalar(8);
    const gradients = tf.grad(a => tf.erf(a.clone()).clone())(a, dy);
    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        await gradients.data(),
        [8 * 2 * Math.exp(-0.5 * 0.5) / Math.sqrt(Math.PI)]);
  });

  it('gradients: Tensor1D', async () => {
    const aValues = [-0.1, 0.2, 0.3, -0.5];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor1d(aValues);
    const dy = tf.tensor1d(dyValues);
    const gradients = tf.grad(a => tf.erf(a))(a, dy);
    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] * 2 * Math.exp(-aValues[i] * aValues[i]) /
          Math.sqrt(Math.PI);
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
    const gradients = tf.grad(a => tf.erf(a))(a, dy);
    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = dyValues[i] * 2 * Math.exp(-aValues[i] * aValues[i]) /
          Math.sqrt(Math.PI);
    }
    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), expected);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.erf({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'erf' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.erf(1);
    expectArraysClose(await result.data(), [0.8427008]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.erf('q'))
        .toThrowError(/Argument 'x' passed to 'erf' must be numeric/);
  });
});
