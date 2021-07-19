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
