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

describeWithFlags('hingeLoss', ALL_ENVS, () => {
  it('1D', async () => {
    const predictions = tf.tensor1d([0, 0, 1, 1]);
    const label = tf.tensor1d([0, 1, 0, 1]);

    const y = tf.losses.hingeLoss(label, predictions);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 1.0);
  });

  it('1D - weighted - Reduction.SUM_BY_NONZERO_WEIGHTS', async () => {
    const predictions = tf.tensor1d([0, 0, 1, 1]);
    const label = tf.tensor1d([0, 1, 0, 1]);
    const weights = tf.tensor1d([0.1, 0.2, 0.3, 0.4]);

    const y = tf.losses.hingeLoss(label, predictions, weights);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 0.225);
  });

  it('1D - weighted - Reduction.NONE', async () => {
    const predictions = tf.tensor1d([0, 0, 1, 1]);
    const label = tf.tensor1d([0, 1, 0, 1]);
    const weights = tf.tensor1d([0.1, 0.2, 0.3, 0.4]);

    const y =
        tf.losses.hingeLoss(label, predictions, weights, tf.Reduction.NONE);

    expect(y.shape).toEqual([4]);
    expectArraysClose(await y.data(), [0.1, 0.2, 0.6, 0.0]);
  });

  it('1D - Reduction.MEAN', async () => {
    const predictions = tf.tensor1d([0, 0, 1, 1]);
    const label = tf.tensor1d([0, 1, 0, 1]);

    const y =
        tf.losses.hingeLoss(label, predictions, undefined, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 1.0);
  });

  it('1D - weighted - Reduction.MEAN', async () => {
    const predictions = tf.tensor1d([0, 0, 1, 1]);
    const label = tf.tensor1d([0, 1, 0, 1]);
    const weights = tf.tensor1d([0.1, 0.2, 0.3, 0.4]);

    const y =
        tf.losses.hingeLoss(label, predictions, weights, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 0.9);
  });

  it('2D', async () => {
    const predictions = tf.tensor2d([0, 0, 0, 1, 1, 1], [2, 3]);
    const label = tf.tensor2d([0, 1, 0, 1, 0, 1], [2, 3]);

    const y = tf.losses.hingeLoss(label, predictions);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 0.8333333);
  });

  it('2D - weighted - Reduction.SUM_BY_NONZERO_WEIGHTS', async () => {
    const predictions = tf.tensor2d([0, 0, 0, 1, 1, 1], [2, 3]);
    const label = tf.tensor2d([0, 1, 0, 1, 0, 1], [2, 3]);
    const weights = tf.tensor2d([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [2, 3]);

    const y = tf.losses.hingeLoss(label, predictions, weights);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 0.26666668);
  });

  it('2D - weighted - Reduction.NONE', async () => {
    const predictions = tf.tensor2d([0, 0, 0, 1, 1, 1], [2, 3]);
    const label = tf.tensor2d([0, 1, 0, 1, 0, 1], [2, 3]);
    const weights = tf.tensor2d([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [2, 3]);

    const y =
        tf.losses.hingeLoss(label, predictions, weights, tf.Reduction.NONE);

    expect(y.shape).toEqual([2, 3]);
    expectArraysClose(await y.data(), [0.1, 0.2, 0.3, 0, 1, 0]);
  });

  it('2D - Reduction.MEAN', async () => {
    const predictions = tf.tensor2d([0, 0, 0, 1, 1, 1], [2, 3]);
    const label = tf.tensor2d([0, 1, 0, 1, 0, 1], [2, 3]);

    const y =
        tf.losses.hingeLoss(label, predictions, undefined, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 0.8333333);
  });

  it('2D - weighted - Reduction.MEAN', async () => {
    const predictions = tf.tensor2d([0, 0, 0, 1, 1, 1], [2, 3]);
    const label = tf.tensor2d([0, 1, 0, 1, 0, 1], [2, 3]);
    const weights = tf.tensor2d([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [2, 3]);

    const y =
        tf.losses.hingeLoss(label, predictions, weights, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 0.76190484);
  });

  it('throws when passed label as a non-tensor', () => {
    const predictions = tf.tensor2d([1, 0, 1, 0, 1, 0], [2, 3]);
    const weights = tf.tensor2d([1, 0, 1, 0, 1, 0], [2, 3]);

    const e = /Argument 'labels' passed to 'hingeLoss' must be a Tensor/;
    expect(
        () => tf.losses.hingeLoss(
            {} as tf.Tensor, predictions, weights, tf.Reduction.MEAN))
        .toThrowError(e);
  });

  it('throws when passed label as a non-tensor', () => {
    const label = tf.tensor2d([1, 0, 1, 0, 1, 0], [2, 3]);
    const weights = tf.tensor2d([1, 0, 1, 0, 1, 0], [2, 3]);

    const e = new RegExp(
        'Argument \'predictions\' passed to \'hingeLoss\' ' +
        'must be a Tensor');
    expect(
        () => tf.losses.hingeLoss(
            label, {} as tf.Tensor, weights, tf.Reduction.MEAN))
        .toThrowError(e);
  });

  it('throws when passed weights as a non-tensor', () => {
    const predictions = tf.tensor2d([1, 0, 1, 0, 1, 0], [2, 3]);
    const label = tf.tensor2d([1, 0, 1, 0, 1, 0], [2, 3]);

    const e = /Argument 'weights' passed to 'hingeLoss' must be a Tensor/;
    expect(
        () => tf.losses.hingeLoss(
            label, predictions, {} as tf.Tensor, tf.Reduction.MEAN))
        .toThrowError(e);
  });

  it('accepts a tensor-like object', async () => {
    const predictions = [0, 0, 1, 1];
    const label = [0, 1, 0, 1];
    const weights = [0.1, 0.2, 0.3, 0.4];

    const y =
        tf.losses.hingeLoss(label, predictions, weights, tf.Reduction.NONE);

    expect(y.shape).toEqual([4]);
    expectArraysClose(await y.data(), [0.1, 0.2, 0.6, 0.0]);
  });
});
