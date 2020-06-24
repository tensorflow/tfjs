/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

describeWithFlags('relu', ALL_ENVS, () => {
  it('basic', async () => {
    const a = tf.tensor1d([1, -2, 0, 3, -0.1]);
    const result = tf.relu(a);
    expectArraysClose(await result.data(), [1, 0, 0, 3, 0]);
  });

  it('5D', async () => {
    const a = tf.tensor5d([1, -2, 5, -3], [1, 2, 2, 1, 1]);
    const result = tf.relu(a);
    expectArraysClose(await result.data(), [1, 0, 5, 0]);
  });

  it('6D', async () => {
    const a = tf.tensor6d([1, -2, 5, -3, -1, 4, 7, 8], [1, 2, 2, 2, 1, 1]);
    const result = tf.relu(a);
    expectArraysClose(await result.data(), [1, 0, 5, 0, 0, 4, 7, 8]);
  });

  it('does nothing to positive values', async () => {
    const a = tf.scalar(1);
    const result = tf.relu(a);
    expectArraysClose(await result.data(), [1]);
  });

  it('sets negative values to 0', async () => {
    const a = tf.scalar(-1);
    const result = tf.relu(a);
    expectArraysClose(await result.data(), [0]);
  });

  it('preserves zero values', async () => {
    const a = tf.scalar(0);
    const result = tf.relu(a);
    expectArraysClose(await result.data(), [0]);
  });

  it('propagates NaNs, float32', async () => {
    const a = tf.tensor1d([1, -2, 0, 3, -0.1, NaN]);
    const result = tf.relu(a);
    expect(result.dtype).toBe('float32');
    expectArraysClose(await result.data(), [1, 0, 0, 3, 0, NaN]);
  });

  it('gradients: positive scalar', async () => {
    const a = tf.scalar(3);
    const dy = tf.scalar(5);

    const grad = tf.grad(a => tf.relu(a));
    const da = grad(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [5]);
  });

  it('gradient with clones', async () => {
    const a = tf.scalar(3);
    const dy = tf.scalar(5);

    const grad = tf.grad(a => tf.relu(a.clone()).clone());
    const da = grad(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [5]);
  });

  it('gradients: negative scalar', async () => {
    const a = tf.scalar(-3);
    const dy = tf.scalar(5);

    const grad = tf.grad(a => tf.relu(a));
    const da = grad(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [0]);
  });

  it('gradients: array', async () => {
    const a = tf.tensor2d([1, -1, 0, .1], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const grad = tf.grad(a => tf.relu(a));
    const da = grad(a, dy);

    expect(da.shape).toEqual(a.shape);
    expect(da.dtype).toEqual('float32');
    expectArraysClose(await da.data(), [1, 0, 0, 4]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.relu({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'relu' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.relu([1, -2, 0, 3, -0.1]);
    expectArraysClose(await result.data(), [1, 0, 0, 3, 0]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.relu('q'))
        .toThrowError(/Argument 'x' passed to 'relu' must be numeric/);
  });
});
