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

describeWithFlags('computeWeightedLoss', ALL_ENVS, () => {
  it('1D - no weights', async () => {
    const losses = tf.tensor1d([1, 2, 3]);

    const y = tf.losses.computeWeightedLoss(losses);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), (1 + 2 + 3) / 3);
  });

  it('1D - no weights - Reduction.NONE', async () => {
    const losses = tf.tensor1d([1, 2, 3]);

    const y =
        tf.losses.computeWeightedLoss(losses, undefined, tf.Reduction.NONE);

    expect(y.shape).toEqual([3]);
    expectArraysClose(await y.data(), [1, 2, 3]);
  });

  it('1D - no weights - Reduction.MEAN', async () => {
    const losses = tf.tensor1d([1, 2, 3]);

    const y =
        tf.losses.computeWeightedLoss(losses, undefined, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), (1 + 2 + 3) / 3);
  });

  it('1D - no weights - Reduction.SUM', async () => {
    const losses = tf.tensor1d([1, 2, 3]);

    const y =
        tf.losses.computeWeightedLoss(losses, undefined, tf.Reduction.SUM);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), (1 + 2 + 3));
  });

  it('1D - weights', async () => {
    const losses = tf.tensor1d([1, 2, 3]);
    const weights = tf.tensor1d([0.1, 0, 0.3]);

    const y = tf.losses.computeWeightedLoss(losses, weights);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), (1 * 0.1 + 2 * 0 + 3 * 0.3) / 2);
  });

  it('2D - weights - broadcast', async () => {
    const losses = tf.tensor2d([[0, 0, 1], [1, 0, 0], [0, 1, 0]], [3, 3]);
    const weights = tf.tensor2d([[0.1, 0.2, 0.3]]);

    const y = tf.losses.computeWeightedLoss(losses, weights);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 0.06666667);
  });

  it('1D - weights - Reduction.NONE', async () => {
    const losses = tf.tensor1d([1, 2, 3]);
    const weights = tf.tensor1d([0.1, 0.2, 0.3]);

    const y = tf.losses.computeWeightedLoss(losses, weights, tf.Reduction.NONE);

    expect(y.shape).toEqual([3]);
    expectArraysClose(await y.data(), [1 * 0.1, 2 * 0.2, 3 * 0.3]);
  });

  it('1D - weights - Reduction.MEAN', async () => {
    const losses = tf.tensor1d([1, 2, 3]);
    const weights = tf.tensor1d([0.1, 0.2, 0.3]);

    const y = tf.losses.computeWeightedLoss(losses, weights, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), (1 * 0.1 + 2 * 0.2 + 3 * 0.3) / 0.6);
  });

  it('1D - weights - Reduction.SUM', async () => {
    const losses = tf.tensor1d([1, 2, 3]);
    const weights = tf.tensor1d([0.1, 0.2, 0.3]);

    const y = tf.losses.computeWeightedLoss(losses, weights, tf.Reduction.SUM);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), (1 * 0.1 + 2 * 0.2 + 3 * 0.3));
  });

  it('2D - no weights', async () => {
    const losses = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);

    const y = tf.losses.computeWeightedLoss(losses);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), (4 + 8 + 12 + 8 + 1 + 3) / 6);
  });

  it('2D - weights', async () => {
    const losses = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);
    const weights = tf.tensor2d([1, 0, 2, -5, 0, 6], [2, 3]);

    const y = tf.losses.computeWeightedLoss(losses, weights);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(),
        (4 * 1 + 8 * 0 + 12 * 2 + (8 * -5) + 1 * 0 + 3 * 6) / 4);
  });

  it('2D - no weights - Reduction.MEAN', async () => {
    const losses = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);

    const y =
        tf.losses.computeWeightedLoss(losses, undefined, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), (4 + 8 + 12 + 8 + 1 + 3) / 6);
  });

  it('2D - weights - Reduction.MEAN', async () => {
    const losses = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);
    const weights = tf.tensor2d([1, 0, 2, -5, 0, 6], [2, 3]);

    const y = tf.losses.computeWeightedLoss(losses, weights, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(),
        (4 * 1 + 8 * 0 + 12 * 2 + (8 * -5) + 1 * 0 + 3 * 6) / 4);
  });

  it('2D - weights - broadcast - MEAN', async () => {
    const losses = tf.tensor2d([[0, 0, 1], [1, 0, 0], [0, 1, 0]], [3, 3]);
    const weights = tf.tensor2d([[0.1, 0.2, 0.3]]);

    const y = tf.losses.computeWeightedLoss(losses, weights, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), (0.3 + 0.1 + 0.2) / (3 * 0.6));
  });

  it('2D - no weights - Reduction.SUM', async () => {
    const losses = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);

    const y =
        tf.losses.computeWeightedLoss(losses, undefined, tf.Reduction.SUM);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), (4 + 8 + 12 + 8 + 1 + 3));
  });

  it('2D - weights - Reduction.SUM', async () => {
    const losses = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);
    const weights = tf.tensor2d([1, 0, 2, -5, 0, 6], [2, 3]);

    const y = tf.losses.computeWeightedLoss(losses, weights, tf.Reduction.SUM);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(), (4 * 1 + 8 * 0 + 12 * 2 + (8 * -5) + 1 * 0 + 3 * 6));
  });

  it('2D - no weights - Reduction.NONE', async () => {
    const losses = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);

    const y =
        tf.losses.computeWeightedLoss(losses, undefined, tf.Reduction.NONE);

    expect(y.shape).toEqual([2, 3]);
    expectArraysClose(await y.data(), [4, 8, 12, 8, 1, 3]);
  });

  it('2D - weights - Reduction.NONE', async () => {
    const losses = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);
    const weights = tf.tensor2d([1, 0, 2, -5, 0, 6], [2, 3]);

    const y = tf.losses.computeWeightedLoss(losses, weights, tf.Reduction.NONE);

    expect(y.shape).toEqual([2, 3]);
    expectArraysClose(
        await y.data(), [4 * 1, 8 * 0, 12 * 2, (8 * -5), 1 * 0, 3 * 6]);
  });

  it('throws when passed losses as a non-tensor', () => {
    const weights = tf.tensor2d([1, 0, 2, -5, 0, 6], [2, 3]);

    const e =
        /Argument 'losses' passed to 'computeWeightedLoss' must be a Tensor/;
    expect(
        () => tf.losses.computeWeightedLoss(
            {} as tf.Tensor, weights, tf.Reduction.NONE))
        .toThrowError(e);
  });

  it('throws when passed weights as a non-tensor', () => {
    const losses = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);

    const e =
        /Argument 'weights' passed to 'computeWeightedLoss' must be a Tensor/;
    expect(
        () => tf.losses.computeWeightedLoss(
            losses, {} as tf.Tensor, tf.Reduction.NONE))
        .toThrowError(e);
  });

  it('accepts a tensor-like object', async () => {
    const losses = [1, 2, 3];
    const weights = [0.1, 0, 0.3];
    const y = tf.losses.computeWeightedLoss(losses, weights);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), (1 * 0.1 + 2 * 0 + 3 * 0.3) / 2);
  });
});
