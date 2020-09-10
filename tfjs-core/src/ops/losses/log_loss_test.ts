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

describeWithFlags('logLoss', ALL_ENVS, () => {
  it('1D', async () => {
    const labels = tf.tensor1d([1, 2, 3]);
    const predictions = tf.tensor1d([0.3, 0.6, 0.1]);

    const y = tf.losses.logLoss(labels, predictions);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 2.668788);
  });

  it('1D - Check for negative values', async () => {
    const labels = tf.tensor1d([1, 2, 3]);
    const predictions = tf.tensor1d([0.3, -0.6, -0.1]);

    const y = tf.losses.logLoss(labels, predictions);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), NaN);
  });

  it('1D - weighted - Reduction.SUM_BY_NONZERO_WEIGHTS', async () => {
    const labels = tf.tensor1d([1, 2, 3]);
    const predictions = tf.tensor1d([0.3, 0.6, 0.1]);
    const weights = tf.tensor1d([0.1, 0.2, 0.3]);

    const y = tf.losses.logLoss(labels, predictions, weights);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 0.7168596);
  });

  it('1D - weighted - Reduction.NONE', async () => {
    const labels = tf.tensor1d([1, 2, 3]);
    const predictions = tf.tensor1d([0.3, 0.6, 0.1]);
    const weights = tf.tensor1d([0.1, 0.2, 0.3]);

    const y = tf.losses.logLoss(
        labels, predictions, weights, undefined, tf.Reduction.NONE);

    expect(y.shape).toEqual([3]);
    expectArraysClose(await y.data(), [0.12039725, 0.02107204, 2.0091095]);
  });

  it('1D - Reduction.MEAN', async () => {
    const labels = tf.tensor1d([1, 2, 3]);
    const predictions = tf.tensor1d([0.3, 0.6, 0.1]);

    const y = tf.losses.logLoss(
        labels, predictions, undefined, undefined, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 2.668788);
  });

  it('1D - weighted - Reduction.MEAN', async () => {
    const labels = tf.tensor1d([1, 2, 3]);
    const predictions = tf.tensor1d([0.3, 0.6, 0.1]);
    const weights = tf.tensor1d([0.1, 0.2, 0.3]);

    const y = tf.losses.logLoss(
        labels, predictions, weights, undefined, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 3.5842977);
  });

  it('2D', async () => {
    const labels = tf.tensor2d([0.4, 0.8, 0.12, 0.8, 0.1, 0.3], [2, 3]);
    const predictions = tf.tensor2d([0.1, 0.7, 0.1, 0.5, 0.05, 0.15], [2, 3]);

    const y = tf.losses.logLoss(labels, predictions);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 0.60019904);
  });

  it('2D - weighted - Reduction.SUM_BY_NONZERO_WEIGHTS', async () => {
    const labels = tf.tensor2d([0.4, 0.8, 0.12, 0.8, 0.1, 0.3], [2, 3]);
    const predictions = tf.tensor2d([0.1, 0.7, 0.1, 0.5, 0.05, 0.15], [2, 3]);
    const weights = tf.tensor2d([3, 0, 5, 0, 4, 2], [2, 3]);

    const y = tf.losses.logLoss(labels, predictions, weights);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 1.8866577);
  });

  it('2D - weighted - Reduction.NONE', async () => {
    const labels = tf.tensor2d([0.4, 0.8, 0.12, 0.8, 0.1, 0.3], [2, 3]);
    const predictions = tf.tensor2d([0.1, 0.7, 0.1, 0.5, 0.05, 0.15], [2, 3]);
    const weights = tf.tensor2d([3, 0, 5, 0, 4, 2], [2, 3]);

    const y = tf.losses.logLoss(
        labels, predictions, weights, undefined, tf.Reduction.NONE);

    expect(y.shape).toEqual([2, 3]);
    expectArraysClose(
        await y.data(), [2.9527497, 0., 1.8451363, 0., 1.3829476, 1.3657978]);
  });

  it('2D - Reduction.MEAN', async () => {
    const labels = tf.tensor2d([0.4, 0.8, 0.12, 0.8, 0.1, 0.3], [2, 3]);
    const predictions = tf.tensor2d([0.1, 0.7, 0.1, 0.5, 0.05, 0.15], [2, 3]);

    const y = tf.losses.logLoss(
        labels, predictions, undefined, undefined, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 0.60019904);
  });

  it('2D - weighted - Reduction.MEAN', async () => {
    const labels = tf.tensor2d([0.4, 0.8, 0.12, 0.8, 0.1, 0.3], [2, 3]);
    const predictions = tf.tensor2d([0.1, 0.7, 0.1, 0.5, 0.05, 0.15], [2, 3]);
    const weights = tf.tensor2d([3, 0, 5, 0, 4, 2], [2, 3]);

    const y = tf.losses.logLoss(
        labels, predictions, weights, undefined, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 0.53904504);
  });

  it('throws when passed label as a non-tensor', () => {
    const predictions = tf.tensor2d([0.1, 0.7, 0.1, 0.5, 0.05, 0.15], [2, 3]);
    const weights = tf.tensor2d([3, 6, 5, 0, 4, 2], [2, 3]);

    const e = /Argument 'labels' passed to 'logLoss' must be a Tensor/;
    expect(
        () => tf.losses.logLoss(
            {} as tf.Tensor, predictions, weights, tf.Reduction.MEAN))
        .toThrowError(e);
  });

  it('throws when passed label as a non-tensor', () => {
    const labels = tf.tensor2d([0.4, 0.8, 0.12, 0.8, 0.1, 0.3], [2, 3]);
    const weights = tf.tensor2d([3, 6, 5, 0, 4, 2], [2, 3]);

    const e = new RegExp(
        'Argument \'predictions\' passed to \'logLoss\' ' +
        'must be a Tensor');
    expect(
        () => tf.losses.logLoss(
            labels, {} as tf.Tensor, weights, tf.Reduction.MEAN))
        .toThrowError(e);
  });

  it('throws when passed weights as a non-tensor', () => {
    const labels = tf.tensor2d([0.4, 0.8, 0.12, 0.8, 0.1, 0.3], [2, 3]);
    const predictions = tf.tensor2d([0.1, 0.7, 0.1, 0.5, 0.05, 0.15], [2, 3]);

    const e = /Argument 'weights' passed to 'logLoss' must be a Tensor/;
    expect(
        () => tf.losses.logLoss(
            labels, predictions, {} as tf.Tensor, tf.Reduction.MEAN))
        .toThrowError(e);
  });

  it('accepts a tensor-like object', async () => {
    const labels = [1, 2, 3];
    const predictions = [0.3, 0.6, 0.1];
    const weights = [0.1, 0.2, 0.3];

    const y = tf.losses.logLoss(
        labels, predictions, weights, undefined, tf.Reduction.NONE);

    expect(y.shape).toEqual([3]);
    expectArraysClose(await y.data(), [0.12039725, 0.02107204, 2.0091095]);
  });
});
