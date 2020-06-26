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

import * as selu_util from './selu_util';

describeWithFlags('selu', ALL_ENVS, () => {
  const scaleAlpha = selu_util.SELU_SCALEALPHA;
  const scale = selu_util.SELU_SCALE;

  it('calculate selu', async () => {
    const a = tf.tensor1d([1, -1, 0]);
    const result = tf.selu(a);

    expect(result.shape).toEqual(a.shape);
    expectArraysClose(await result.data(), [1.0507, -1.1113, 0]);
  });

  it('selu propagates NaN', async () => {
    const a = tf.tensor1d([1, NaN]);
    const result = tf.selu(a);
    expect(result.shape).toEqual(a.shape);
    expectArraysClose(await result.data(), [1.0507, NaN]);
  });

  it('gradients: Scalar', async () => {
    let aValue = 1;
    let dyValue = 1;
    let a = tf.scalar(aValue);
    let dy = tf.scalar(dyValue);

    let gradients = tf.grad(a => tf.selu(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [dyValue * scale]);

    aValue = -1;
    dyValue = 2;
    a = tf.scalar(aValue);
    dy = tf.scalar(dyValue);

    gradients = tf.grad(a => tf.selu(a))(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(
        await gradients.data(), [dyValue * scaleAlpha * Math.exp(aValue)]);
  });

  it('gradient with clones', async () => {
    const aValue = 1;
    const dyValue = 1;
    const a = tf.scalar(aValue);
    const dy = tf.scalar(dyValue);

    const gradients = tf.grad(a => tf.selu(a.clone()).clone())(a, dy);

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), [dyValue * scale]);
  });

  it('gradients: Tensor1D', async () => {
    const aValues = [1, -1, 0];
    const dyValues = [1, 2, 3];
    const a = tf.tensor1d(aValues);
    const dy = tf.tensor1d(dyValues);

    const gradients = tf.grad(a => tf.selu(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      if (aValues[i] > 0) {
        expected[i] = dyValues[i] * scale;
      } else {
        expected[i] = dyValues[i] * scaleAlpha * Math.exp(aValues[i]);
      }
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), expected);
  });

  it('gradients: Tensor2D', async () => {
    const aValues = [1, -1, 0, 0.5];
    const dyValues = [1, 2, 3, 4];
    const a = tf.tensor2d(aValues, [2, 2]);
    const dy = tf.tensor2d(dyValues, [2, 2]);

    const gradients = tf.grad(a => tf.selu(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      if (aValues[i] > 0) {
        expected[i] = dyValues[i] * scale;
      } else {
        expected[i] = dyValues[i] * scaleAlpha * Math.exp(aValues[i]);
      }
    }

    expect(gradients.shape).toEqual(a.shape);
    expect(gradients.dtype).toEqual('float32');
    expectArraysClose(await gradients.data(), expected);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.selu({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'selu' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.selu([1, -1, 0]);
    expect(result.shape).toEqual([3]);
    expectArraysClose(await result.data(), [1.0507, -1.1113, 0]);
  });

  it('throws for string tensor', () => {
    expect(() => tf.selu('q'))
        .toThrowError(/Argument 'x' passed to 'selu' must be numeric/);
  });
});
