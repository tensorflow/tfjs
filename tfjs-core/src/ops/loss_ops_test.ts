/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

describeWithFlags('meanSquaredError', ALL_ENVS, () => {
  it('1D', async () => {
    const predictions = tf.tensor1d([1, 2, 3]);
    const label = tf.tensor1d([0.3, -0.6, -0.1]);

    const y = tf.losses.meanSquaredError(label, predictions);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(),
        ((1 - 0.3) * (1 - 0.3) + (2 - (-0.6)) * (2 - (-0.6)) +
         (3 - (-0.1)) * (3 - (-0.1))) /
            3);
  });

  it('1D - weighted - Reduction.SUM_BY_NONZERO_WEIGHTS', async () => {
    const predictions = tf.tensor1d([1, 2, 3]);
    const label = tf.tensor1d([0.3, -0.6, -0.1]);
    const weights = tf.tensor1d([0.1, 0.2, 0.3]);

    const y = tf.losses.meanSquaredError(label, predictions, weights);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(),
        ((1 - 0.3) * (1 - 0.3) * 0.1 + (2 - (-0.6)) * (2 - (-0.6)) * 0.2 +
         (3 - (-0.1)) * (3 - (-0.1)) * 0.3) /
            3);
  });

  it('1D - weighted - Reduction.NONE', async () => {
    const predictions = tf.tensor1d([1, 2, 3]);
    const label = tf.tensor1d([0.3, -0.6, -0.1]);
    const weights = tf.tensor1d([0.1, 0.2, 0.3]);

    const y = tf.losses.meanSquaredError(
        label, predictions, weights, tf.Reduction.NONE);

    expect(y.shape).toEqual([3]);
    expectArraysClose(await y.data(), [
      (1 - 0.3) * (1 - 0.3) * 0.1, (2 - (-0.6)) * (2 - (-0.6)) * 0.2,
      (3 - (-0.1)) * (3 - (-0.1)) * 0.3
    ]);
  });

  it('1D - Reduction.MEAN', async () => {
    const predictions = tf.tensor1d([1, 2, 3]);
    const label = tf.tensor1d([0.3, -0.6, -0.1]);

    const y = tf.losses.meanSquaredError(
        label, predictions, undefined, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(),
        ((1 - 0.3) * (1 - 0.3) + (2 - (-0.6)) * (2 - (-0.6)) +
         (3 - (-0.1)) * (3 - (-0.1))) /
            3);
  });

  it('1D - weighted - Reduction.MEAN', async () => {
    const predictions = tf.tensor1d([1, 2, 3]);
    const label = tf.tensor1d([0.3, -0.6, -0.1]);
    const weights = tf.tensor1d([0.1, 0.2, 0.3]);

    const y = tf.losses.meanSquaredError(
        label, predictions, weights, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(),
        (((1 - 0.3) * (1 - 0.3) * 0.1) + ((2 - (-0.6)) * (2 - (-0.6)) * 0.2) +
         ((3 - (-0.1)) * (3 - (-0.1)) * 0.3)) /
            0.6);
  });

  it('2D', async () => {
    const predictions = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);
    const label = tf.tensor2d([1, 9, 2, -5, -2, 6], [2, 3]);

    const y = tf.losses.meanSquaredError(label, predictions);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(),
        ((4 - 1) * (4 - 1) + (8 - 9) * (8 - 9) + (12 - 2) * (12 - 2) +
         (8 - (-5)) * (8 - (-5)) + (1 - (-2)) * (1 - (-2)) +
         (3 - 6) * (3 - 6)) /
            6);
  });

  it('2D - weighted - Reduction.SUM_BY_NONZERO_WEIGHTS', async () => {
    const predictions = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);
    const label = tf.tensor2d([1, 9, 2, -5, -2, 6], [2, 3]);
    const weights = tf.tensor2d([3, 0, 5, 0, 4, 2], [2, 3]);

    const y = tf.losses.meanSquaredError(label, predictions, weights);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(),
        ((4 - 1) * (4 - 1) * 3 + (8 - 9) * (8 - 9) * 0 +
         (12 - 2) * (12 - 2) * 5 + (8 - (-5)) * (8 - (-5)) * 0 +
         (1 - (-2)) * (1 - (-2)) * 4 + (3 - 6) * (3 - 6) * 2) /
            4);
  });

  it('2D - weighted - Reduction.NONE', async () => {
    const predictions = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);
    const label = tf.tensor2d([1, 9, 2, -5, -2, 6], [2, 3]);
    const weights = tf.tensor2d([3, 6, 5, 0, 4, 2], [2, 3]);

    const y = tf.losses.meanSquaredError(
        label, predictions, weights, tf.Reduction.NONE);

    expect(y.shape).toEqual([2, 3]);
    expectArraysClose(await y.data(), [
      (4 - 1) * (4 - 1) * 3, (8 - 9) * (8 - 9) * 6, (12 - 2) * (12 - 2) * 5,
      (8 - (-5)) * (8 - (-5)) * 0, (1 - (-2)) * (1 - (-2)) * 4,
      (3 - 6) * (3 - 6) * 2
    ]);
  });

  it('2D - Reduction.MEAN', async () => {
    const predictions = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);
    const label = tf.tensor2d([1, 9, 2, -5, -2, 6], [2, 3]);

    const y = tf.losses.meanSquaredError(
        label, predictions, undefined, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(),
        ((4 - 1) * (4 - 1) + (8 - 9) * (8 - 9) + (12 - 2) * (12 - 2) +
         (8 - (-5)) * (8 - (-5)) + (1 - (-2)) * (1 - (-2)) +
         (3 - 6) * (3 - 6)) /
            6);
  });

  it('2D - weighted - Reduction.MEAN', async () => {
    const predictions = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);
    const label = tf.tensor2d([1, 9, 2, -5, -2, 6], [2, 3]);
    const weights = tf.tensor2d([3, 6, 5, 0, 4, 2], [2, 3]);

    const y = tf.losses.meanSquaredError(
        label, predictions, weights, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(),
        ((4 - 1) * (4 - 1) * 3 + (8 - 9) * (8 - 9) * 6 +
         (12 - 2) * (12 - 2) * 5 + (8 - (-5)) * (8 - (-5)) * 0 +
         (1 - (-2)) * (1 - (-2)) * 4 + (3 - 6) * (3 - 6) * 2) /
            20);
  });

  it('throws when passed label as a non-tensor', () => {
    const predictions = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);
    const weights = tf.tensor2d([3, 6, 5, 0, 4, 2], [2, 3]);

    const e = /Argument 'labels' passed to 'meanSquaredError' must be a Tensor/;
    expect(
        () => tf.losses.meanSquaredError(
            {} as tf.Tensor, predictions, weights, tf.Reduction.MEAN))
        .toThrowError(e);
  });

  it('throws when passed label as a non-tensor', () => {
    const label = tf.tensor2d([1, 9, 2, -5, -2, 6], [2, 3]);
    const weights = tf.tensor2d([3, 6, 5, 0, 4, 2], [2, 3]);

    const e = new RegExp(
        'Argument \'predictions\' passed to \'meanSquaredError\' ' +
        'must be a Tensor');
    expect(
        () => tf.losses.meanSquaredError(
            label, {} as tf.Tensor, weights, tf.Reduction.MEAN))
        .toThrowError(e);
  });

  it('throws when passed weights as a non-tensor', () => {
    const predictions = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);
    const label = tf.tensor2d([1, 9, 2, -5, -2, 6], [2, 3]);

    const e =
        /Argument 'weights' passed to 'meanSquaredError' must be a Tensor/;
    expect(
        () => tf.losses.meanSquaredError(
            label, predictions, {} as tf.Tensor, tf.Reduction.MEAN))
        .toThrowError(e);
  });

  it('accepts a tensor-like object', async () => {
    const predictions = [1, 2, 3];
    const label = [0.3, -0.6, -0.1];
    const weights = [0.1, 0.2, 0.3];

    const y = tf.losses.meanSquaredError(
        label, predictions, weights, tf.Reduction.NONE);

    expect(y.shape).toEqual([3]);
    expectArraysClose(await y.data(), [
      (1 - 0.3) * (1 - 0.3) * 0.1, (2 - (-0.6)) * (2 - (-0.6)) * 0.2,
      (3 - (-0.1)) * (3 - (-0.1)) * 0.3
    ]);
  });
});

describeWithFlags('cosineDistance', ALL_ENVS, () => {
  it('1D', async () => {
    const predictions = tf.tensor1d([1, 2, 3]);
    const label = tf.tensor1d([0.3, -0.6, -0.1]);

    const y = tf.losses.cosineDistance(label, predictions, 0);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 1 - (1 * 0.3 + 2 * -0.6 + 3 * -0.1));
  });

  it('1D - weighted - Reduction.SUM_BY_NONZERO_WEIGHTS', async () => {
    const predictions = tf.tensor1d([1, 2, 3]);
    const label = tf.tensor1d([0.3, -0.6, -0.1]);
    const weights = tf.scalar(0.1);

    const y = tf.losses.cosineDistance(label, predictions, 0, weights);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(), (1 - (1 * 0.3 + 2 * -0.6 + 3 * -0.1)) * 0.1);
  });

  it('1D - weighted - Reduction.NONE', async () => {
    const predictions = tf.tensor1d([1, 2, 3]);
    const label = tf.tensor1d([0.3, -0.6, -0.1]);
    const weights = tf.scalar(0.1);

    const y = tf.losses.cosineDistance(
        label, predictions, 0, weights, tf.Reduction.NONE);

    expect(y.shape).toEqual([1]);
    expectArraysClose(
        await y.data(), [(1 - (1 * 0.3 + 2 * -0.6 + 3 * -0.1)) * 0.1]);
  });

  it('1D - Reduction.MEAN', async () => {
    const predictions = tf.tensor1d([1, 2, 3]);
    const label = tf.tensor1d([0.3, -0.6, -0.1]);

    const y = tf.losses.cosineDistance(
        label, predictions, 0, undefined, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), (1 - (1 * 0.3 + 2 * -0.6 + 3 * -0.1)));
  });

  it('1D - weighted - Reduction.MEAN', async () => {
    const predictions = tf.tensor1d([1, 2, 3]);
    const label = tf.tensor1d([0.3, -0.6, -0.1]);
    const weights = tf.scalar(0.1);

    const y = tf.losses.cosineDistance(
        label, predictions, 0, weights, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(), ((1 - (1 * 0.3 + 2 * -0.6 + 3 * -0.1)) * 0.1) / 0.1);
  });

  it('2D', async () => {
    const predictions = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);
    const label = tf.tensor2d([1, 9, 2, -5, -2, 6], [2, 3]);

    const y = tf.losses.cosineDistance(label, predictions, 1);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(),
        ((1 - (4 * 1 + 8 * 9 + 12 * 2)) + (1 - (8 * -5 + 1 * -2 + 3 * 6))) / 2);
  });

  it('2D - weighted - Reduction.SUM_BY_NONZERO_WEIGHTS', async () => {
    const predictions = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);
    const label = tf.tensor2d([1, 9, 2, -5, -2, 6], [2, 3]);
    const weights = tf.tensor2d([3, 0], [2, 1]);

    const y = tf.losses.cosineDistance(label, predictions, 1, weights);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(),
        ((1 - (4 * 1 + 8 * 9 + 12 * 2)) * 3 +
         (1 - (8 * -5 + 1 * -2 + 3 * 6)) * 0) /
            1);
  });

  it('2D - weighted - Reduction.NONE', async () => {
    const predictions = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);
    const label = tf.tensor2d([1, 9, 2, -5, -2, 6], [2, 3]);
    const weights = tf.tensor2d([3, 0], [2, 1]);

    const y = tf.losses.cosineDistance(
        label, predictions, 1, weights, tf.Reduction.NONE);

    expect(y.shape).toEqual([2, 1]);
    expectArraysClose(await y.data(), [
      (1 - (4 * 1 + 8 * 9 + 12 * 2)) * 3, (1 - (8 * -5 + 1 * -2 + 3 * 6)) * 0
    ]);
  });

  it('2D - Reduction.MEAN', async () => {
    const predictions = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);
    const label = tf.tensor2d([1, 9, 2, -5, -2, 6], [2, 3]);

    const y = tf.losses.cosineDistance(
        label, predictions, 1, undefined, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(),
        ((1 - (4 * 1 + 8 * 9 + 12 * 2)) + (1 - (8 * -5 + 1 * -2 + 3 * 6))) / 2);
  });

  it('2D - weighted - Reduction.MEAN', async () => {
    const predictions = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);
    const label = tf.tensor2d([1, 9, 2, -5, -2, 6], [2, 3]);
    const weights = tf.tensor2d([3, 0], [2, 1]);

    const y = tf.losses.cosineDistance(
        label, predictions, 1, weights, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(),
        ((1 - (4 * 1 + 8 * 9 + 12 * 2)) * 3 +
         (1 - (8 * -5 + 1 * -2 + 3 * 6)) * 0) /
            3);
  });

  it('throws when passed label as a non-tensor', () => {
    const predictions = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);
    const weights = tf.tensor2d([3, 6, 5, 0, 4, 2], [2, 3]);

    const e = /Argument 'labels' passed to 'cosineDistance' must be a Tensor/;
    expect(
        () => tf.losses.cosineDistance(
            {} as tf.Tensor, predictions, 0, weights, tf.Reduction.MEAN))
        .toThrowError(e);
  });

  it('throws when passed label as a non-tensor', () => {
    const label = tf.tensor2d([1, 9, 2, -5, -2, 6], [2, 3]);
    const weights = tf.tensor2d([3, 6, 5, 0, 4, 2], [2, 3]);

    const e = new RegExp(
        'Argument \'predictions\' passed to \'cosineDistance\' ' +
        'must be a Tensor');
    expect(
        () => tf.losses.cosineDistance(
            label, {} as tf.Tensor, 0, weights, tf.Reduction.MEAN))
        .toThrowError(e);
  });

  it('throws when passed weights as a non-tensor', () => {
    const predictions = tf.tensor2d([4, 8, 12, 8, 1, 3], [2, 3]);
    const label = tf.tensor2d([1, 9, 2, -5, -2, 6], [2, 3]);

    const e = /Argument 'weights' passed to 'cosineDistance' must be a Tensor/;
    expect(
        () => tf.losses.cosineDistance(
            label, predictions, 0, {} as tf.Tensor, tf.Reduction.MEAN))
        .toThrowError(e);
  });

  it('accepts a tensor-like object', async () => {
    const predictions = [1, 2, 3];
    const label = [0.3, -0.6, -0.1];
    const weights = 0.1;

    const y = tf.losses.cosineDistance(
        label, predictions, 0, weights, tf.Reduction.NONE);

    expect(y.shape).toEqual([1]);
    expectArraysClose(
        await y.data(), [(1 - (1 * 0.3 + 2 * -0.6 + 3 * -0.1)) * 0.1]);
  });
});

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

describeWithFlags('huberLoss', ALL_ENVS, () => {
  it('1D', async () => {
    const labels = tf.tensor1d([1, 2, 3]);
    const predictions = tf.tensor1d([0.3, 0.6, 0.1]);

    const y = tf.losses.huberLoss(labels, predictions);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 1.1816667);
  });

  it('1D - delta', async () => {
    const labels = tf.tensor1d([1, 2, 3]);
    const predictions = tf.tensor1d([0.3, 0.6, 0.1]);
    const delta = 0.4;

    const y = tf.losses.huberLoss(labels, predictions, undefined, delta);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 0.58666664);
  });

  it('1D - weighted - Reduction.SUM_BY_NONZERO_WEIGHTS', async () => {
    const labels = tf.tensor1d([1, 2, 3]);
    const predictions = tf.tensor1d([0.3, 0.6, 0.1]);
    const weights = tf.tensor1d([0.1, 0.2, 0.3]);

    const y = tf.losses.huberLoss(labels, predictions, weights);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 0.30816665);
  });

  it('1D - weighted - Reduction.NONE', async () => {
    const labels = tf.tensor1d([1, 2, 3]);
    const predictions = tf.tensor1d([0.3, 0.6, 0.1]);
    const weights = tf.tensor1d([0.1, 0.2, 0.3]);

    const y = tf.losses.huberLoss(
        labels, predictions, weights, undefined, tf.Reduction.NONE);

    expect(y.shape).toEqual([3]);
    expectArraysClose(await y.data(), [0.0245, 0.17999999, 0.72]);
  });

  it('1D - Reduction.MEAN', async () => {
    const labels = tf.tensor1d([1, 2, 3]);
    const predictions = tf.tensor1d([0.3, 0.6, 0.1]);

    const y = tf.losses.huberLoss(
        labels, predictions, undefined, undefined, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 1.1816667);
  });

  it('1D - weighted - Reduction.MEAN', async () => {
    const labels = tf.tensor1d([1, 2, 3]);
    const predictions = tf.tensor1d([0.3, 0.6, 0.1]);
    const weights = tf.tensor1d([0.1, 0.2, 0.3]);

    const y = tf.losses.huberLoss(
        labels, predictions, weights, undefined, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 1.5408332);
  });

  it('2D', async () => {
    const labels = tf.tensor2d([0.4, 0.8, 0.12, 0.8, 0.1, 0.3], [2, 3]);
    const predictions = tf.tensor2d([0.1, 0.7, 0.1, 0.5, 0.05, 0.15], [2, 3]);

    const y = tf.losses.huberLoss(labels, predictions);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 0.01795);
  });

  it('2D - weighted - Reduction.SUM_BY_NONZERO_WEIGHTS', async () => {
    const labels = tf.tensor2d([0.4, 0.8, 0.12, 0.8, 0.1, 0.3], [2, 3]);
    const predictions = tf.tensor2d([0.1, 0.7, 0.1, 0.5, 0.05, 0.15], [2, 3]);
    const weights = tf.tensor2d([3, 0, 5, 0, 4, 2], [2, 3]);

    const y = tf.losses.huberLoss(labels, predictions, weights);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 0.040875003);
  });

  it('2D - weighted - Reduction.NONE', async () => {
    const labels = tf.tensor2d([0.4, 0.8, 0.12, 0.8, 0.1, 0.3], [2, 3]);
    const predictions = tf.tensor2d([0.1, 0.7, 0.1, 0.5, 0.05, 0.15], [2, 3]);
    const weights = tf.tensor2d([3, 0, 5, 0, 4, 2], [2, 3]);

    const y = tf.losses.huberLoss(
        labels, predictions, weights, undefined, tf.Reduction.NONE);

    expect(y.shape).toEqual([2, 3]);
    expectArraysClose(await y.data(), [0.135, 0., 0.001, 0., 0.005, 0.0225]);
  });

  it('2D - Reduction.MEAN', async () => {
    const labels = tf.tensor2d([0.4, 0.8, 0.12, 0.8, 0.1, 0.3], [2, 3]);
    const predictions = tf.tensor2d([0.1, 0.7, 0.1, 0.5, 0.05, 0.15], [2, 3]);

    const y = tf.losses.huberLoss(
        labels, predictions, undefined, undefined, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 0.01795);
  });

  it('2D - weighted - Reduction.MEAN', async () => {
    const labels = tf.tensor2d([0.4, 0.8, 0.12, 0.8, 0.1, 0.3], [2, 3]);
    const predictions = tf.tensor2d([0.1, 0.7, 0.1, 0.5, 0.05, 0.15], [2, 3]);
    const weights = tf.tensor2d([3, 0, 5, 0, 4, 2], [2, 3]);

    const y = tf.losses.huberLoss(
        labels, predictions, weights, undefined, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 0.011678572);
  });

  it('throws when passed label as a non-tensor', () => {
    const predictions = tf.tensor2d([0.1, 0.7, 0.1, 0.5, 0.05, 0.15], [2, 3]);
    const weights = tf.tensor2d([3, 6, 5, 0, 4, 2], [2, 3]);

    const e = /Argument 'labels' passed to 'huberLoss' must be a Tensor/;
    expect(
        () => tf.losses.huberLoss(
            {} as tf.Tensor, predictions, weights, tf.Reduction.MEAN))
        .toThrowError(e);
  });

  it('throws when passed label as a non-tensor', () => {
    const labels = tf.tensor2d([0.4, 0.8, 0.12, 0.8, 0.1, 0.3], [2, 3]);
    const weights = tf.tensor2d([3, 6, 5, 0, 4, 2], [2, 3]);

    const e = new RegExp(
        'Argument \'predictions\' passed to \'huberLoss\' ' +
        'must be a Tensor');
    expect(
        () => tf.losses.huberLoss(
            labels, {} as tf.Tensor, weights, tf.Reduction.MEAN))
        .toThrowError(e);
  });

  it('throws when passed weights as a non-tensor', () => {
    const labels = tf.tensor2d([0.4, 0.8, 0.12, 0.8, 0.1, 0.3], [2, 3]);
    const predictions = tf.tensor2d([0.1, 0.7, 0.1, 0.5, 0.05, 0.15], [2, 3]);

    const e = /Argument 'weights' passed to 'huberLoss' must be a Tensor/;
    expect(
        () => tf.losses.huberLoss(
            labels, predictions, {} as tf.Tensor, tf.Reduction.MEAN))
        .toThrowError(e);
  });

  it('accepts a tensor-like object', async () => {
    const labels = [1, 2, 3];
    const predictions = [0.3, 0.6, 0.1];
    const weights = [0.1, 0.2, 0.3];

    const y = tf.losses.huberLoss(
        labels, predictions, weights, undefined, tf.Reduction.NONE);

    expect(y.shape).toEqual([3]);
    expectArraysClose(await y.data(), [0.0245, 0.17999999, 0.72]);
  });
});

describeWithFlags('sigmoidCrossEntropy', ALL_ENVS, () => {
  it('All wrong', async () => {
    const label = tf.tensor2d([[0, 0, 1], [1, 0, 0], [0, 1, 0]], [3, 3]);
    const predictions = tf.tensor2d(
        [[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]],
        [3, 3]);

    const y = tf.losses.sigmoidCrossEntropy(label, predictions);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 6.6667123);
  });

  it('All right', async () => {
    const label = tf.tensor2d([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [3, 3]);
    const predictions = tf.tensor2d(
        [[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]],
        [3, 3]);

    const y = tf.losses.sigmoidCrossEntropy(label, predictions);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 0);
  });

  it('Weighted - Reduction.SUM_BY_NONZERO_WEIGHTS', async () => {
    const label = tf.tensor2d([[0, 0, 1], [1, 0, 0], [0, 1, 0]], [3, 3]);
    const predictions = tf.tensor2d(
        [[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]],
        [3, 3]);

    const weights = tf.tensor2d([[0.1, 0.2, 0.3]]);

    const y = tf.losses.sigmoidCrossEntropy(label, predictions, weights);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 1.3333424);
  });

  it('Weighted - Reduction.NONE', async () => {
    const label = tf.tensor2d([[0, 0, 1], [1, 0, 0], [0, 1, 0]], [3, 3]);
    const predictions = tf.tensor2d(
        [[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]],
        [3, 3]);
    const weights = tf.tensor2d([[0.1, 0.2, 0.3]]);

    const y = tf.losses.sigmoidCrossEntropy(
        label, predictions, weights, undefined, tf.Reduction.NONE);

    expect(y.shape).toEqual([3, 3]);
    expectArraysClose(await y.data(), [
      1.0000046, 9.0797803e-06, 3.0000138e+00, 1.0000046e+00, 2.0000093e+00,
      1.3619671e-05, 4.5398901e-06, 2.0000093e+00, 3.0000138e+00
    ]);
  });

  it('Reduction.MEAN', async () => {
    const label = tf.tensor2d([[0, 0, 1], [1, 0, 0], [0, 1, 0]], [3, 3]);
    const predictions = tf.tensor2d(
        [[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]],
        [3, 3]);

    const y = tf.losses.sigmoidCrossEntropy(
        label, predictions, undefined, undefined, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 6.6667123);
  });

  it('Weighted - Reduction.MEAN', async () => {
    const label = tf.tensor2d([[0, 0, 1], [1, 0, 0], [0, 1, 0]], [3, 3]);
    const predictions = tf.tensor2d(
        [[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]],
        [3, 3]);
    const weights = tf.tensor2d([[0.1, 0.2, 0.3]]);

    const y = tf.losses.sigmoidCrossEntropy(
        label, predictions, weights, undefined, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(),
        6.666712284088135,
    );
  });

  it('Label Smoothing - Weighted - Reduction.MEAN', async () => {
    const label = tf.tensor2d([[0, 0, 1], [1, 0, 0], [0, 1, 0]], [3, 3]);
    const predictions = tf.tensor2d(
        [[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]],
        [3, 3]);
    const weights = tf.tensor2d([[0.1, 0.2, 0.3]]);
    const labelSmoothing = 0.3;

    const y = tf.losses.sigmoidCrossEntropy(
        label, predictions, weights, labelSmoothing, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 6.1667128);
  });

  it('throws when multiClassLabels and logits are of different shapes', () => {
    const multiClassLabels =
        tf.tensor2d([10, 10, 10, 10, 10, 10, 10, 10, 10], [3, 3]);
    const logits = tf.tensor2d([10, 10, 10, 10, 10, 10], [2, 3]);

    const e = new RegExp(
        'Error in sigmoidCrossEntropy:  Shapes 3,3 and 2,3 must match');
    expect(() => tf.losses.sigmoidCrossEntropy(multiClassLabels, logits))
        .toThrowError(e);
  });

  it('throws when passed multiClassLabels as a non-tensor', () => {
    const predictions = tf.tensor2d(
        [[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]],
        [3, 3]);
    const weights = tf.tensor2d([[0.1, 0.2, 0.3]]);

    const e = new RegExp(
        'Argument \'multiClassLabels\' passed to \'sigmoidCrossEntropy\' ' +
        'must be a Tensor');

    expect(
        () => tf.losses.sigmoidCrossEntropy(
            {} as tf.Tensor, predictions, weights, tf.Reduction.MEAN))
        .toThrowError(e);
  });

  it('throws when passed logits as a non-tensor', () => {
    const label = tf.tensor2d([[0, 0, 1], [1, 0, 0], [0, 1, 0]], [3, 3]);
    const weights = tf.tensor2d([[0.1, 0.2, 0.3]]);

    const e = new RegExp(
        'Argument \'logits\' passed to \'sigmoidCrossEntropy\' ' +
        'must be a Tensor');
    expect(
        () => tf.losses.sigmoidCrossEntropy(
            label, {} as tf.Tensor, weights, tf.Reduction.MEAN))
        .toThrowError(e);
  });

  it('throws when passed weights as a non-tensor', () => {
    const label = tf.tensor2d([[0, 0, 1], [1, 0, 0], [0, 1, 0]], [3, 3]);
    const predictions = tf.tensor2d(
        [[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]],
        [3, 3]);

    const e =
        /Argument 'weights' passed to 'sigmoidCrossEntropy' must be a Tensor/;
    expect(
        () => tf.losses.sigmoidCrossEntropy(
            label, predictions, {} as tf.Tensor, tf.Reduction.MEAN))
        .toThrowError(e);
  });
});

describeWithFlags('softmaxCrossEntropy', ALL_ENVS, () => {
  it('All wrong', async () => {
    const label = tf.tensor2d([[0, 0, 1], [1, 0, 0], [0, 1, 0]], [3, 3]);
    const predictions = tf.tensor2d(
        [[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]],
        [3, 3]);

    const y = tf.losses.softmaxCrossEntropy(label, predictions);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 20);
  });

  it('All right', async () => {
    const label = tf.tensor2d([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [3, 3]);
    const predictions = tf.tensor2d(
        [[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]],
        [3, 3]);

    const y = tf.losses.softmaxCrossEntropy(label, predictions);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 0);
  });

  it('Weighted - Reduction.SUM_BY_NONZERO_WEIGHTS', async () => {
    const label = tf.tensor2d([[0, 0, 1], [1, 0, 0], [0, 1, 0]], [3, 3]);
    const predictions = tf.tensor2d(
        [[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]],
        [3, 3]);

    const weights =
        tf.tensor2d([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]);

    const y = tf.losses.softmaxCrossEntropy(label, predictions, weights);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 4);
  });

  it('Weighted - Reduction.NONE', async () => {
    const label = tf.tensor2d([[0, 0, 1], [1, 0, 0], [0, 1, 0]], [3, 3]);
    const predictions = tf.tensor2d(
        [[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]],
        [3, 3]);
    const weights = tf.tensor1d([0.1, 0.2, 0.3]);

    const y = tf.losses.softmaxCrossEntropy(
        label, predictions, weights, undefined, tf.Reduction.NONE);

    expect(y.shape).toEqual([3]);
    expectArraysClose(await y.data(), [2, 4, 6]);
  });

  it('Reduction.MEAN', async () => {
    const label = tf.tensor2d([[0, 0, 1], [1, 0, 0], [0, 1, 0]], [3, 3]);
    const predictions = tf.tensor2d(
        [[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]],
        [3, 3]);

    const y = tf.losses.softmaxCrossEntropy(
        label, predictions, undefined, undefined, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 20);
  });

  it('Weighted - Reduction.MEAN', async () => {
    const label = tf.tensor2d([[0, 0, 1], [1, 0, 0], [0, 1, 0]], [3, 3]);
    const predictions = tf.tensor2d(
        [[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]],
        [3, 3]);
    const weights = tf.tensor1d([0.1, 0.2, 0.3]);

    const y = tf.losses.softmaxCrossEntropy(
        label, predictions, weights, undefined, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(
        await y.data(),
        20,
    );
  });

  it('Label Smoothing - Weighted - Reduction.MEAN', async () => {
    const label = tf.tensor2d([[0, 0, 1], [1, 0, 0], [0, 1, 0]], [3, 3]);
    const predictions = tf.tensor2d(
        [[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]],
        [3, 3]);
    const weights = tf.tensor2d([[0.1, 0.2, 0.3]]);
    const labelSmoothing = 0.3;

    const y = tf.losses.softmaxCrossEntropy(
        label, predictions, weights, labelSmoothing, tf.Reduction.MEAN);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 18);
  });

  it('throws when multiClassLabels and logits are of different shapes', () => {
    const multiClassLabels =
        tf.tensor2d([10, 10, 10, 10, 10, 10, 10, 10, 10], [3, 3]);
    const logits = tf.tensor2d([10, 10, 10, 10, 10, 10], [2, 3]);

    const e = new RegExp(
        'Error in softmaxCrossEntropy:  Shapes 3,3 and 2,3 must match');
    expect(() => tf.losses.softmaxCrossEntropy(multiClassLabels, logits))
        .toThrowError(e);
  });

  it('throws when passed multiClassLabels as a non-tensor', () => {
    const predictions = tf.tensor2d(
        [[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]],
        [3, 3]);
    const weights = tf.tensor2d([[0.1, 0.2, 0.3]]);

    const e = new RegExp(
        'Argument \'onehotLabels\' passed to \'softmaxCrossEntropy\' ' +
        'must be a Tensor');

    expect(
        () => tf.losses.softmaxCrossEntropy(
            {} as tf.Tensor, predictions, weights, tf.Reduction.MEAN))
        .toThrowError(e);
  });

  it('throws when passed logits as a non-tensor', () => {
    const label = tf.tensor2d([[0, 0, 1], [1, 0, 0], [0, 1, 0]], [3, 3]);
    const weights = tf.tensor2d([[0.1, 0.2, 0.3]]);

    const e = new RegExp(
        'Argument \'logits\' passed to \'softmaxCrossEntropy\' ' +
        'must be a Tensor');
    expect(
        () => tf.losses.softmaxCrossEntropy(
            label, {} as tf.Tensor, weights, tf.Reduction.MEAN))
        .toThrowError(e);
  });

  it('throws when passed weights as a non-tensor', () => {
    const label = tf.tensor2d([[0, 0, 1], [1, 0, 0], [0, 1, 0]], [3, 3]);
    const predictions = tf.tensor2d(
        [[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]],
        [3, 3]);

    const e =
        /Argument 'weights' passed to 'softmaxCrossEntropy' must be a Tensor/;
    expect(
        () => tf.losses.softmaxCrossEntropy(
            label, predictions, {} as tf.Tensor, tf.Reduction.MEAN))
        .toThrowError(e);
  });

  it('accepts a tensor-like object', async () => {
    const label = [[0, 0, 1], [1, 0, 0], [0, 1, 0]];
    const predictions =
        [[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]];

    const y = tf.losses.softmaxCrossEntropy(label, predictions);

    expect(y.shape).toEqual([]);
    expectArraysClose(await y.data(), 20);
  });
});
