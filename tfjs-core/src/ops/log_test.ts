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

describeWithFlags('log', ALL_ENVS, () => {
  it('log', async () => {
    const a = tf.tensor1d([1, 2]);
    const r = tf.log(a);
    expectArraysClose(await r.data(), [Math.log(1), Math.log(2)]);
  });

  it('log 6D', async () => {
    const a = tf.range(1, 65).reshape([2, 2, 2, 2, 2, 2]);
    const r = tf.log(a);

    const expectedResult = [];
    for (let i = 1; i < 65; i++) {
      expectedResult[i - 1] = Math.log(i);
    }

    expectArraysClose(await r.data(), expectedResult);
  });

  it('log propagates NaNs', async () => {
    const a = tf.tensor1d([1, NaN]);
    const r = tf.log(a);
    expectArraysClose(await r.data(), [Math.log(1), NaN]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(5);
    const dy = tf.scalar(3);

    const gradients = tf.grad(a => tf.log(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [3 / 5]);
  });

  it('gradient with clones', async () => {
    const a = tf.scalar(5);
    const dy = tf.scalar(3);

    const gradients = tf.grad(a => tf.log(a.clone()).clone())(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [3 / 5]);
  });

  it('gradients: Tensor1D', async () => {
    const a = tf.tensor1d([-1, 2, 3, -5]);
    const dy = tf.tensor1d([1, 2, 3, 4]);

    const gradients = tf.grad(a => tf.log(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [1 / -1, 2 / 2, 3 / 3, 4 / -5]);
  });

  it('gradients: Tensor2D', async () => {
    const a = tf.tensor2d([-3, 1, 2, 3], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const gradients = tf.grad(a => tf.log(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [1 / -3, 2 / 1, 3 / 2, 4 / 3]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.log({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'log' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const r = tf.log([1, 2]);
    expectArraysClose(await r.data(), [Math.log(1), Math.log(2)]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.log('q'))
        .toThrowError(/Argument 'x' passed to 'log' must be numeric/);
  });
});
