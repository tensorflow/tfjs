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
import {expectArraysClose, expectArraysEqual} from '../test_util';

describeWithFlags('logSumExp', ALL_ENVS, () => {
  it('0', async () => {
    const a = tf.scalar(0);
    const result = tf.logSumExp(a);
    expectArraysClose(await result.data(), 0);
  });

  it('basic', async () => {
    const a = tf.tensor1d([1, 2, -3]);
    const result = tf.logSumExp(a);

    expectArraysClose(
        await result.data(),
        Math.log(Math.exp(1) + Math.exp(2) + Math.exp(-3)));
  });

  it('propagates NaNs', async () => {
    const a = tf.tensor1d([1, 2, NaN]);
    const result = tf.logSumExp(a);
    expectArraysEqual(await result.data(), NaN);
  });

  it('axes=0 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const r = tf.logSumExp(a, [0]);

    expect(r.shape).toEqual([2]);
    const expected = [
      Math.log(Math.exp(1) + Math.exp(3) + Math.exp(0)),
      Math.log(Math.exp(2) + Math.exp(0) + Math.exp(1))
    ];
    expectArraysClose(await r.data(), expected);
  });

  it('axes=0 in 2D array, keepDims', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const r = tf.logSumExp(a, [0], true /* keepDims */);

    expect(r.shape).toEqual([1, 2]);
    const expected = [
      Math.log(Math.exp(1) + Math.exp(3) + Math.exp(0)),
      Math.log(Math.exp(2) + Math.exp(0) + Math.exp(1))
    ];
    expectArraysClose(await r.data(), expected);
  });

  it('axes=1 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.logSumExp(a, [1]);

    expect(res.shape).toEqual([3]);
    const expected = [
      Math.log(Math.exp(1) + Math.exp(2)),
      Math.log(Math.exp(3) + Math.exp(0)),
      Math.log(Math.exp(0) + Math.exp(1)),
    ];
    expectArraysClose(await res.data(), expected);
  });

  it('axes = -1 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.logSumExp(a, -1);

    expect(res.shape).toEqual([3]);
    const expected = [
      Math.log(Math.exp(1) + Math.exp(2)),
      Math.log(Math.exp(3) + Math.exp(0)),
      Math.log(Math.exp(0) + Math.exp(1)),
    ];
    expectArraysClose(await res.data(), expected);
  });

  it('2D, axes=1 provided as a single digit', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [2, 3]);
    const res = tf.logSumExp(a, 1);

    expect(res.shape).toEqual([2]);
    const expected = [
      Math.log(Math.exp(1) + Math.exp(2) + Math.exp(3)),
      Math.log(Math.exp(0) + Math.exp(0) + Math.exp(1))
    ];
    expectArraysClose(await res.data(), expected);
  });

  it('axes=0,1 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const res = tf.logSumExp(a, [0, 1]);

    expect(res.shape).toEqual([]);
    const expected = [Math.log(
        Math.exp(1) + Math.exp(2) + Math.exp(3) + Math.exp(0) + Math.exp(0) +
        Math.exp(1))];
    expectArraysClose(await res.data(), expected);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.logSumExp({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'logSumExp' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const result = tf.logSumExp([1, 2, -3]);
    expectArraysClose(
        await result.data(),
        Math.log(Math.exp(1) + Math.exp(2) + Math.exp(-3)));
  });

  it('throws error for string tensor', () => {
    expect(() => tf.logSumExp(['a']))
        .toThrowError(
            /Argument 'x' passed to 'logSumExp' must be numeric tensor/);
  });
});
