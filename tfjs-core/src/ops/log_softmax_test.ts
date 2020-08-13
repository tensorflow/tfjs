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

describeWithFlags('logSoftmax', ALL_ENVS, () => {
  it('regular test', async () => {
    const y = tf.logSoftmax(tf.tensor1d([2, 1, 3]));

    expectArraysClose(await y.data(), [-1.407606, -2.4076061, -0.407606]);
  });

  it('Huge difference', async () => {
    const y = tf.logSoftmax(tf.tensor1d([-1000, +1000]));

    expectArraysClose(await y.data(), [-2000, 0]);
  });

  it('Propagates NaNs', async () => {
    const a = tf.tensor1d([2, 1, NaN]);
    const y = tf.logSoftmax(a);
    expectArraysClose(await y.data(), [NaN, NaN, NaN]);
  });

  it('2D, axis=1', async () => {
    const y = tf.logSoftmax(tf.tensor2d([[2, 1, 3], [1, 3, 2]], [2, 3]), 1);
    const expected =
        [-1.407606, -2.4076061, -0.407606, -2.4076061, -0.4076061, -1.4076061];
    expect(y.rank).toBe(2);
    expectArraysClose(await y.data(), expected);
  });

  it('2D, implicit axis=1', async () => {
    const y = tf.logSoftmax(tf.tensor2d([[2, 1, 3], [1, 3, 2]], [2, 3]));
    const expected =
        [-1.407606, -2.4076061, -0.407606, -2.4076061, -0.4076061, -1.4076061];
    expect(y.rank).toBe(2);
    expectArraysClose(await y.data(), expected);
  });

  it('1D gradient', async () => {
    const x = tf.tensor1d([1, 2, 10]);
    const dy = tf.tensor1d([1, 2, 3]);
    const dx = tf.grad((x) => x.logSoftmax())(x, dy);

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), [0.9992599, 1.9979881, -2.9972477]);
  });

  it('2D, axis=0 throws error', () => {
    const f = () => {
      tf.logSoftmax(tf.tensor2d([[2, 1, 3], [1, 3, 2]], [2, 3]), 0);
    };
    expect(f).toThrowError();
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.logSoftmax({} as tf.Tensor))
        .toThrowError(
            /Argument 'logits' passed to 'logSoftmax' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const y = tf.logSoftmax([2, 1, 3]);

    expectArraysClose(await y.data(), [-1.407606, -2.4076061, -0.407606]);
  });
});
