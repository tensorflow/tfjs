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

describeWithFlags('logSigmoid', ALL_ENVS, () => {
  it('basic', async () => {
    const values = [1, -3, 2, 7, -4];
    const a = tf.tensor1d(values);

    const result = tf.logSigmoid(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.log(1 / (1 + Math.exp(-values[i])));
    }
    expectArraysClose(await result.data(), expected);
  });

  it('scalar', async () => {
    const a = tf.scalar(-2);

    const result = tf.logSigmoid(a);

    const expected = [Math.log(1 / (1 + Math.exp(2)))];
    expectArraysClose(await result.data(), expected);
  });

  it('tensor2D', async () => {
    const values = [1, 2, -3, 5];
    const a = tf.tensor2d(values, [2, 2]);

    const result = tf.logSigmoid(a);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      expected[i] = Math.log(1 / (1 + Math.exp(-values[i])));
    }
    expectArraysClose(await result.data(), expected);
  });

  it('larger magnitude negative inputs', async () => {
    const values = [-100, -200, -3000];
    const a = tf.tensor1d(values);

    const result = tf.logSigmoid(a);

    const expected = [-100, -200, -3000];

    expectArraysClose(await result.data(), expected);
  });

  it('larger magnitude positive inputs', async () => {
    const values = [100, 200, 3000, 50000];
    const a = tf.tensor1d(values);

    const result = tf.logSigmoid(a);

    const expected = [0, 0, 0, 0];

    expectArraysClose(await result.data(), expected);
  });

  it('propagates NaNs', async () => {
    const a = tf.tensor1d([3, NaN]);
    const res = tf.logSigmoid(a);
    expectArraysClose(
        await res.data(), [Math.log(1 / (1 + Math.exp(-3))), NaN]);
  });

  it('gradients: Scalar', async () => {
    const a = tf.scalar(3);
    const dy = tf.scalar(4);
    const dyVal = await dy.array();

    const da = tf.grad(a => tf.logSigmoid(a))(a, dy);
    const aVal = await a.array();
    const y = 1 / (1 + Math.exp(aVal));
    expectArraysClose(await da.data(), [dyVal * y]);
  });

  it('gradients: Tensor1D', async () => {
    const a = tf.tensor1d([1, 2, -3, 5]);
    const aVals = await a.array();
    const dy = tf.tensor1d([1, 2, 3, 4]);
    const dyVals = await dy.array();
    const da = tf.grad(a => tf.logSigmoid(a))(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      const y = 1 / (1 + Math.exp(aVals[i]));
      expected[i] = dyVals[i] * y;
    }

    expectArraysClose(await da.data(), expected);
  });

  it('gradient with clones', async () => {
    const a = tf.tensor1d([1, 2, -3, 5]);
    const aVals = await a.array();
    const dy = tf.tensor1d([1, 2, 3, 4]);
    const dyVals = await dy.array();
    const da = tf.grad(a => tf.logSigmoid(a.clone()).clone())(a, dy);

    const expected = [];
    for (let i = 0; i < a.size; i++) {
      const y = 1 / (1 + Math.exp(aVals[i]));
      expected[i] = dyVals[i] * y;
    }

    expectArraysClose(await da.data(), expected);
  });

  it('gradients: Tensor2D', async () => {
    const a = tf.tensor2d([1, 2, -3, 5], [2, 2]);
    const dy = tf.tensor2d([1, 2, 3, 4], [2, 2]);

    const da = tf.grad(a => tf.logSigmoid(a))(a, dy);

    const expected = [];
    const aVals = await a.data();
    const dyVals = await dy.data();
    for (let i = 0; i < a.size; i++) {
      const y = 1 / (1 + Math.exp(aVals[i]));
      expected[i] = dyVals[i] * y;
    }

    expectArraysClose(await da.data(), expected);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.logSigmoid({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'logSigmoid' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.logSigmoid(-2);
    const expected = [Math.log(1 / (1 + Math.exp(2)))];
    expectArraysClose(await result.data(), expected);
  });

  it('throws for string tensor', () => {
    expect(() => tf.logSigmoid('q'))
        .toThrowError(/Argument 'x' passed to 'logSigmoid' must be numeric/);
  });
});
