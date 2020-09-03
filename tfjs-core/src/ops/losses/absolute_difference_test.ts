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

import * as tf from '../../index';
import {ALL_ENVS, describeWithFlags} from '../../jasmine_util';
import {expectArraysClose} from '../../test_util';

describeWithFlags('absoluteDifference', ALL_ENVS, () => {
  it('1D', async () => {
    const predictions = tf.tensor1d([1, 2, 3]);
    const label = tf.tensor1d([0.3, -0.6, -0.1]);

    const y = tf.losses.absoluteDifference(label, predictions);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(),
        (Math.abs(1 - 0.3) + Math.abs(2 - (-0.6)) + Math.abs(3 - (-0.1))) / 3);
  });

  it('1D - weighted - Reduction.SUM_BY_NONZERO_WEIGHTS', async () => {
    const predictions = tf.tensor1d([1, 2, 3]);
    const label = tf.tensor1d([0.3, -0.6, -0.1]);
    const weights = tf.tensor1d([0.1, 0.2, 0.3]);

    const y = tf.losses.absoluteDifference(label, predictions, weights);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(),
        (Math.abs(1 - 0.3) * 0.1 + Math.abs(2 - (-0.6)) * 0.2 +
         Math.abs(3 - (-0.1)) * 0.3) /
            3);
  });

  it('1D - weighted - Reduction.NONE', async () => {
    const predictions = tf.tensor1d([1, 2, 3]);
    const label = tf.tensor1d([0.3, -0.6, -0.1]);
    const weights = tf.tensor1d([0.1, 0.2, 0.3]);

    const y = tf.losses.absoluteDifference(
        label, predictions, weights, tf.Reduction.NONE);

    expect(y.shape).toEqual([3]);
    expectArraysClose(await y.data(), [
      Math.abs(1 - 0.3) * 0.1, Math.abs(2 - (-0.6)) * 0.2,
      Math.abs(3 - (-0.1)) * 0.3
    ]);
  });

  it('1D - Reduction.MEAN', async () => {
    const predictions = tf.tensor1d([1, 2, 3]);
    const label = tf.tensor1d([0.3, -0.6, -0.1]);

    const y = tf.losses.absoluteDifference(
        label, predictions, undefined, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(),
        (Math.abs(1 - 0.3) + Math.abs(2 - (-0.6)) + Math.abs(3 - (-0.1))) / 3);
  });

  it('1D - weighted - Reduction.MEAN', async () => {
    const predictions = tf.tensor1d([1, 2, 3]);
    const label = tf.tensor1d([0.3, -0.6, -0.1]);
    const weights = tf.tensor1d([0.1, 0.2, 0.3]);

    const y = tf.losses.absoluteDifference(
        label, predictions, weights, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(),
        ((Math.abs(1 - 0.3) * 0.1) + (Math.abs(2 - (-0.6)) * 0.2) +
         (Math.abs(3 - (-0.1)) * 0.3)) /
            0.6);
  });

  it('2D', async () => {
    const predictions = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);
    const label = tf.tensor2d([1, 9, 2, -5, -2, 6], [2, 3]);

    const y = tf.losses.absoluteDifference(label, predictions);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(),
        (Math.abs(4 - 1) + Math.abs(8 - 9) + Math.abs(12 - 2) +
         Math.abs(8 - (-5)) + Math.abs(1 - (-2)) + Math.abs(3 - 6)) /
            6);
  });

  it('2D - weighted - Reduction.SUM_BY_NONZERO_WEIGHTS', async () => {
    const predictions = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);
    const label = tf.tensor2d([1, 9, 2, -5, -2, 6], [2, 3]);
    const weights = tf.tensor2d([3, 0, 5, 0, 4, 2], [2, 3]);

    const y = tf.losses.absoluteDifference(label, predictions, weights);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(),
        (Math.abs(4 - 1) * 3 + Math.abs(8 - 9) * 0 + Math.abs(12 - 2) * 5 +
         Math.abs(8 - (-5)) * 0 + Math.abs(1 - (-2)) * 4 +
         Math.abs(3 - 6) * 2) /
            4);
  });

  it('2D - weighted - Reduction.NONE', async () => {
    const predictions = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);
    const label = tf.tensor2d([1, 9, 2, -5, -2, 6], [2, 3]);
    const weights = tf.tensor2d([3, 6, 5, 0, 4, 2], [2, 3]);

    const y = tf.losses.absoluteDifference(
        label, predictions, weights, tf.Reduction.NONE);

    expect(y.shape).toEqual([2, 3]);
    expectArraysClose(await y.data(), [
      Math.abs(4 - 1) * 3, Math.abs(8 - 9) * 6, Math.abs(12 - 2) * 5,
      Math.abs(8 - (-5)) * 0, Math.abs(1 - (-2)) * 4, Math.abs(3 - 6) * 2
    ]);
  });

  it('2D - Reduction.MEAN', async () => {
    const predictions = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);
    const label = tf.tensor2d([1, 9, 2, -5, -2, 6], [2, 3]);

    const y = tf.losses.absoluteDifference(
        label, predictions, undefined, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(),
        (Math.abs(4 - 1) + Math.abs(8 - 9) + Math.abs(12 - 2) +
         Math.abs(8 - (-5)) + Math.abs(1 - (-2)) + Math.abs(3 - 6)) /
            6);
  });

  it('2D - weighted - Reduction.MEAN', async () => {
    const predictions = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);
    const label = tf.tensor2d([1, 9, 2, -5, -2, 6], [2, 3]);
    const weights = tf.tensor2d([3, 6, 5, 0, 4, 2], [2, 3]);

    const y = tf.losses.absoluteDifference(
        label, predictions, weights, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(),
        (Math.abs(4 - 1) * 3 + Math.abs(8 - 9) * 6 + Math.abs(12 - 2) * 5 +
         Math.abs(8 - (-5)) * 0 + Math.abs(1 - (-2)) * 4 +
         Math.abs(3 - 6) * 2) /
            20);
  });

  it('throws when passed label as a non-tensor', () => {
    const predictions = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);
    const weights = tf.tensor2d([3, 6, 5, 0, 4, 2], [2, 3]);

    const e =
        /Argument 'labels' passed to 'absoluteDifference' must be a Tensor/;
    expect(
        () => tf.losses.absoluteDifference(
            {} as tf.Tensor, predictions, weights, tf.Reduction.MEAN))
        .toThrowError(e);
  });

  it('throws when passed label as a non-tensor', () => {
    const label = tf.tensor2d([1, 9, 2, -5, -2, 6], [2, 3]);
    const weights = tf.tensor2d([3, 6, 5, 0, 4, 2], [2, 3]);

    const e = new RegExp(
        'Argument \'predictions\' passed to \'absoluteDifference\' ' +
        'must be a Tensor');
    expect(
        () => tf.losses.absoluteDifference(
            label, {} as tf.Tensor, weights, tf.Reduction.MEAN))
        .toThrowError(e);
  });

  it('throws when passed weights as a non-tensor', () => {
    const predictions = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);
    const label = tf.tensor2d([1, 9, 2, -5, -2, 6], [2, 3]);

    const e =
        /Argument 'weights' passed to 'absoluteDifference' must be a Tensor/;
    expect(
        () => tf.losses.absoluteDifference(
            label, predictions, {} as tf.Tensor, tf.Reduction.MEAN))
        .toThrowError(e);
  });

  it('accepts a tensor-like object', async () => {
    const predictions = [1, 2, 3];
    const label = [0.3, -0.6, -0.1];
    const weights = [0.1, 0.2, 0.3];

    const y = tf.losses.absoluteDifference(
        label, predictions, weights, tf.Reduction.NONE);

    expect(y.shape).toEqual([3]);
    expectArraysClose(await y.data(), [
      Math.abs(1 - 0.3) * 0.1, Math.abs(2 - (-0.6)) * 0.2,
      Math.abs(3 - (-0.1)) * 0.3
    ]);
  });
});
