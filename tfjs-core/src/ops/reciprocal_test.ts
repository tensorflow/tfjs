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

describeWithFlags('reciprocal', ALL_ENVS, () => {
  it('1D array', async () => {
    const a = tf.tensor1d([2, 3, 0, NaN]);
    const r = tf.reciprocal(a);
    expectArraysClose(await r.data(), [1 / 2, 1 / 3, Infinity, NaN]);
  });

  it('2D array', async () => {
    const a = tf.tensor2d([1, Infinity, 0, NaN], [2, 2]);
    const r = tf.reciprocal(a);
    expect(r.shape).toEqual([2, 2]);
    expectArraysClose(await r.data(), [1 / 1, 0, Infinity, NaN]);
  });

  it('reciprocal propagates NaNs', async () => {
    const a = tf.tensor1d([1.5, NaN]);
    const r = tf.reciprocal(a);
    expectArraysClose(await r.data(), [1 / 1.5, NaN]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.reciprocal(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [-1 * 8 * (1 / (5 * 5))]);
  });

  it('gradient with clones', async () => {
    const a = tf.scalar(5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.reciprocal(a.clone()).clone())(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [-1 * 8 * (1 / (5 * 5))]);
  });

  it('gradients: Tensor1D', async () => {
    const a = tf.tensor1d([-1, 2, 3, -5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.reciprocal(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [
      -1 * 1 * (1 / (-1 * -1)), -1 * 2 * (1 / (2 * 2)), -1 * 3 * (1 / (3 * 3)),
      -1 * 4 * (1 / (-5 * -5))
    ]);
  });

  it('gradients: Tensor2D', async () => {
    const a = tf.tensor2d([-1, 2, 3, -5], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.reciprocal(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [
      -1 * 1 * (1 / (-1 * -1)), -1 * 2 * (1 / (2 * 2)), -1 * 3 * (1 / (3 * 3)),
      -1 * 4 * (1 / (-5 * -5))
    ]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.reciprocal({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'reciprocal' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const r = tf.reciprocal([2, 3, 0, NaN]);
    expectArraysClose(await r.data(), [1 / 2, 1 / 3, Infinity, NaN]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.reciprocal('q'))
        .toThrowError(/Argument 'x' passed to 'reciprocal' must be numeric/);
  });
});
