/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

describeWithFlags('softmax', ALL_ENVS, () => {
  it('regular test', async () => {
    const y = tf.softmax(tf.tensor1d([2, 1, 3]));

    expectArraysClose(await y.data(), [0.24472847, 0.09003057, 0.66524095]);
    expectArraysClose(await y.sum().data(), 1);
  });

  it('overflow', async () => {
    const y = tf.softmax(tf.tensor1d([100, 100]));

    expectArraysClose(await y.data(), [0.5, 0.5]);
  });

  it('underflow', async () => {
    const y = tf.softmax(tf.tensor1d([-100, -100]));

    expectArraysClose(await y.data(), [0.5, 0.5]);
  });

  it('Huge difference between probabilities', async () => {
    const y = tf.softmax(tf.tensor1d([-1000, +1000]));

    expectArraysClose(await y.data(), [0, 1]);
  });

  it('Propagates NaNs', async () => {
    const a = tf.tensor1d([2, 1, NaN]);
    const y = tf.softmax(a);
    expectArraysClose(await y.data(), [NaN, NaN, NaN]);
  });

  it('2D, dim=1', async () => {
    const y = tf.softmax(tf.tensor2d([[2, 1, 3], [1, 3, 2]], [2, 3]), 1);
    const expected = [
      0.24472847, 0.09003057, 0.66524095, 0.09003057, 0.66524095, 0.24472847
    ];
    expect(y.rank).toBe(2);
    expectArraysClose(await y.data(), expected);
  });

  it('2D, implicit dim=1', async () => {
    const y = tf.softmax(tf.tensor2d([[2, 1, 3], [1, 3, 2]], [2, 3]));
    const expected = [
      0.24472847, 0.09003057, 0.66524095, 0.09003057, 0.66524095, 0.24472847
    ];
    expect(y.rank).toBe(2);
    expectArraysClose(await y.data(), expected);
  });

  it('2D, dim=0 throws error', () => {
    const f = () => {
      tf.softmax(tf.tensor2d([[2, 1, 3], [1, 3, 2]], [2, 3]), 0);
    };
    expect(f).toThrowError();
  });

  it('1D gradient', async () => {
    const x = tf.tensor1d([10, 0, -1]);
    const y = tf.softmax(x);
    const dy = tf.tensor1d([1, 2, 3]);
    const dx = tf.grad((x) => x.softmax())(x, dy);

    const totalSum = tf.sum(tf.mul(dy, y)) as tf.Scalar;

    const dyVals = await dy.array();
    const sumVals = await totalSum.array();
    const yVals = await y.array();
    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), [
      (dyVals[0] - sumVals) * yVals[0],
      (dyVals[1] - sumVals) * yVals[1],
      (dyVals[2] - sumVals) * yVals[2],
    ]);
  });

  it('gradient with clones', () => {
    const x = tf.tensor1d([10, 0, -1]);
    const dx = tf.grad((x) => x.clone().softmax().clone())(x);
    expect(dx.shape).toEqual(x.shape);
    expect(dx.dtype).toBe('float32');
  });

  it('2D gradient', async () => {
    const x = tf.tensor2d([10, 0, -1, 5, 4, 3], [2, 3]);
    const y = tf.softmax(x);
    const dy = tf.tensor2d([3, 2, 1, 1, 2, 3], [2, 3]);
    const dx = tf.grad((x) => x.softmax())(x, dy);

    const axis = -1;
    const totalSum = tf.sum(tf.mulStrict(dy, y), axis) as tf.Tensor1D;

    const dyVals = await dy.array();
    const sumVals = await totalSum.array();
    const yVals = await y.array();

    expect(dx.shape).toEqual(x.shape);
    expectArraysClose(await dx.data(), [
      (dyVals[0][0] - sumVals[0]) * yVals[0][0],
      (dyVals[0][1] - sumVals[0]) * yVals[0][1],
      (dyVals[0][2] - sumVals[0]) * yVals[0][2],
      (dyVals[1][0] - sumVals[1]) * yVals[1][0],
      (dyVals[1][1] - sumVals[1]) * yVals[1][1],
      (dyVals[1][2] - sumVals[1]) * yVals[1][2]
    ]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.softmax({} as tf.Tensor))
        .toThrowError(/Argument 'logits' passed to 'softmax' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const y = tf.softmax([2, 1, 3]);

    expectArraysClose(await y.data(), [0.24472847, 0.09003057, 0.66524095]);
    expectArraysClose(await y.sum().data(), 1);
  });
});

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
