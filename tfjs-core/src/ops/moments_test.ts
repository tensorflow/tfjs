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

describeWithFlags('moments', ALL_ENVS, () => {
  it('basic', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const {mean, variance} = tf.moments(a);

    expect(mean.dtype).toBe('float32');
    expect(variance.dtype).toBe('float32');
    expectArraysClose(await mean.data(), 7 / 6);
    expectArraysClose(await variance.data(), 1.1389);
  });

  it('propagates NaNs', async () => {
    const a = tf.tensor2d([1, 2, 3, NaN, 0, 1], [3, 2]);
    const {mean, variance} = tf.moments(a);

    expect(mean.dtype).toBe('float32');
    expect(variance.dtype).toBe('float32');
    expectArraysEqual(await mean.data(), NaN);
    expectArraysEqual(await variance.data(), NaN);
  });

  it('moments(int32) => float32', async () => {
    const a = tf.tensor1d([1, 5, 7, 3], 'int32');
    const {mean, variance} = tf.moments(a);

    expect(mean.dtype).toBe('float32');
    expect(variance.dtype).toBe('float32');
    expectArraysClose(await mean.data(), 4);
    expectArraysClose(await variance.data(), 5);
  });

  it('moments(bool) => float32', async () => {
    const a = tf.tensor1d([true, false, false, true, true], 'bool');
    const {mean, variance} = tf.moments(a);

    expect(mean.dtype).toBe('float32');
    expect(variance.dtype).toBe('float32');
    expectArraysClose(await mean.data(), 3 / 5);
    expectArraysClose(await variance.data(), 0.23999998);
  });

  it('2D array with keep dim', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const {mean, variance} = tf.moments(a, null, true /* keepDims */);

    expect(mean.shape).toEqual([1, 1]);
    expect(mean.dtype).toBe('float32');
    expect(variance.shape).toEqual([1, 1]);
    expect(variance.dtype).toBe('float32');
    expectArraysClose(await mean.data(), [7 / 6]);
    expectArraysClose(await variance.data(), [1.138889]);
  });

  it('axis=0 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const {mean, variance} = tf.moments(a, [0]);

    expect(mean.shape).toEqual([2]);
    expect(mean.dtype).toBe('float32');
    expect(variance.shape).toEqual([2]);
    expect(variance.dtype).toBe('float32');
    expectArraysClose(await mean.data(), [4 / 3, 1]);
    expectArraysClose(await variance.data(), [1.556, 2 / 3]);
  });

  it('axis=1 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const {mean, variance} = tf.moments(a, [1]);

    expect(mean.dtype).toBe('float32');
    expect(mean.shape).toEqual([3]);
    expect(variance.dtype).toBe('float32');
    expect(variance.shape).toEqual([3]);
    expectArraysClose(await mean.data(), [1.5, 1.5, 0.5]);
    expectArraysClose(await variance.data(), [0.25, 2.25, 0.25]);
  });

  it('2D, axis=1 provided as number', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [2, 3]);
    const {mean, variance} = tf.moments(a, 1);

    expect(mean.shape).toEqual([2]);
    expect(mean.dtype).toBe('float32');
    expect(variance.shape).toEqual([2]);
    expect(variance.dtype).toBe('float32');
    expectArraysClose(await mean.data(), [2, 1 / 3]);
    expectArraysClose(await variance.data(), [2 / 3, 0.222]);
  });

  it('2D, axis=-1 provided as number', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [2, 3]);
    const {mean, variance} = tf.moments(a, -1);

    expect(mean.shape).toEqual([2]);
    expect(mean.dtype).toBe('float32');
    expect(variance.shape).toEqual([2]);
    expect(variance.dtype).toBe('float32');
    expectArraysClose(await mean.data(), [2, 1 / 3]);
    expectArraysClose(await variance.data(), [2 / 3, 0.222]);
  });

  it('axis=0,1 in 2D array', async () => {
    const a = tf.tensor2d([1, 2, 3, 0, 0, 1], [3, 2]);
    const {mean, variance} = tf.moments(a, [0, 1]);

    expect(mean.shape).toEqual([]);
    expect(mean.dtype).toBe('float32');
    expect(variance.shape).toEqual([]);
    expect(variance.dtype).toBe('float32');
    expectArraysClose(await mean.data(), [7 / 6]);
    expectArraysClose(await variance.data(), [1.1389]);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.moments({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'moments' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const {mean, variance} = tf.moments([1, 2, 3, 0, 0, 1]);

    expect(mean.dtype).toBe('float32');
    expect(variance.dtype).toBe('float32');
    expectArraysClose(await mean.data(), 7 / 6);
    expectArraysClose(await variance.data(), 1.1389);
  });
});
