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

describeWithFlags('elu', ALL_ENVS, () => {
  it('calculate elu', async () => {
    const a = tf.tensor1d([1, -1, 0]);
    const result = tf.elu(a);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(await result.data(), [1, -0.6321, 0]);
  });

  it('elu propagates NaN', async () => {
    const a = tf.tensor1d([1, NaN]);
    const result = tf.elu(a);
    expect(result.shape).toEqual(a.shape);
    expectArraysClose(await result.data(), [1, NaN]);
  });

  it('derivative', async () => {
    const x = tf.tensor1d([1, 3, -2]);
    const dy = tf.tensor1d([5, 50, 500]);
    const gradients = tf.grad(a => tf.elu(a))(x, dy);

    expect(gradients.shape).toEqual(x.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [5, 50, 500 * Math.exp(-2)]);
  });

  it('gradient with clones', async () => {
    const x = tf.tensor1d([1, 3, -2]);
    const dy = tf.tensor1d([5, 50, 500]);
    const gradients = tf.grad(a => tf.elu(a.clone()).clone())(x, dy);

    expect(gradients.shape).toEqual(x.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [5, 50, 500 * Math.exp(-2)]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.elu({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'elu' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.elu([1, -1, 0]);
    expect(result.shape).toEqual(result.shape);
    expectArraysClose(await result.data(), [1, -0.6321, 0]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.elu('q'))
        .toThrowError(/Argument 'x' passed to 'elu' must be numeric/);
  });
});
