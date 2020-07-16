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

describeWithFlags('square', ALL_ENVS, () => {
  it('1D array', async () => {
    const a = tf.tensor1d([2, 4, Math.sqrt(2)]);
    const r = tf.square(a);
    expectArraysClose(await r.data(), [4, 16, 2]);
  });

  it('2D array', async () => {
    const a = tf.tensor2d([1, 2, Math.sqrt(2), Math.sqrt(3)], [2, 2]);
    const r = tf.square(a);
    expect(r.shape).toEqual([2, 2]);
    expectArraysClose(await r.data(), [1, 4, 2, 3]);
  });

  it('5D array', async () => {
    const a = tf.tensor5d([1, 2, Math.sqrt(2), Math.sqrt(3)], [1, 1, 2, 2, 1]);
    const r = tf.square(a);
    expect(r.shape).toEqual([1, 1, 2, 2, 1]);
    expectArraysClose(await r.data(), [1, 4, 2, 3]);
  });

  it('6D array', async () => {
    const a = tf.tensor6d(
        [1, 2, Math.sqrt(2), Math.sqrt(3), 3, 4, Math.sqrt(7), Math.sqrt(13)],
        [1, 1, 2, 2, 2, 1]);
    const r = tf.square(a);
    expect(r.shape).toEqual(a.shape);
    expectArraysClose(await r.data(), [1, 4, 2, 3, 9, 16, 7, 13]);
  });

  it('square propagates NaNs', async () => {
    const a = tf.tensor1d([1.5, NaN]);
    const r = tf.square(a);
    expectArraysClose(await r.data(), [2.25, NaN]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.square(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [2 * 5 * 8]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(5);
    const dy = tf.scalar(8);

    const gradients = tf.grad(a => tf.square(a.clone()).clone())(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [2 * 5 * 8]);
  });

  it('gradients: Tensor1D', async () => {
    const a = tf.tensor1d([-1, 2, 3, -5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.square(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [-2, 4 * 2, 6 * 3, -10 * 4]);
  });

  it('gradients: Tensor2D', async () => {
    const a = tf.tensor2d([-3, 1, 2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.square(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [-6 * 1, 2 * 2, 4 * 3, 6 * 4]);
  });

  it('gradients: Tensor5D', async () => {
    const a = tf.tensor5d([-3, 1, 2, 3], [1, 1, 1, 2, 2]);
    const dy = tf.tensor5d([1, 2, 3, 4], [1, 1, 1, 2, 2]);

    const gradients = tf.grad(a => tf.square(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [-6 * 1, 2 * 2, 4 * 3, 6 * 4]);
  });

  it('gradients: Tensor6D', async () => {
    const a = tf.tensor6d([-3, 1, 2, 3, -4, 5, 12, 3], [1, 1, 1, 2, 2, 2]);
    const dy = tf.tensor6d([1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 1, 2, 2, 2]);

    const gradients = tf.grad(a => tf.square(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        await gradients.data(),
        [-6 * 1, 2 * 2, 4 * 3, 6 * 4, -8 * 5, 10 * 6, 24 * 7, 6 * 8]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.square({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'square' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const r = tf.square([2, 4, Math.sqrt(2)]);
    expectArraysClose(await r.data(), [4, 16, 2]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.square('q'))
        .toThrowError(/Argument 'x' passed to 'square' must be numeric/);
  });
});
