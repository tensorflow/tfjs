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

describeWithFlags('AdamaxOptimizer', ALL_ENVS, () => {
  it('basic', async () => {
    const learningRate = 0.1;
    const beta1 = 0.8;
    const beta2 = 0.9;
    const decay = 0.1;
    const optimizer =
        tf.train.adamax(learningRate, beta1, beta2, undefined, decay);

    const x = tf.tensor1d([2, 4]).variable();

    const f = () => x.square().sum() as tf.Scalar;

    let numTensors = tf.memory().numTensors;

    let cost = optimizer.minimize(f, /* returnCost */ true);

    // Cost & 2 accumulators should be the only additional arrays.
    expect(tf.memory().numTensors).toBe(numTensors + 3);
    // new_first_m = [
    //    beta1 * old_first_m_w1 + (1-beta1) * grad_w1,
    //    beta1 * old_first_m_w2 + (1-beta1) * grad_w2
    // ] = [.8, 1.6]
    //
    // ut_0 = beta2 * old_weighted_inf_norm = [0, 0]
    // u1_1 = [
    //    abs(grad_w1),
    //    abs(grad_w2)
    // ] = [4, 8]
    // new_weighted_inf_norm = max(ut_0, ut_1) = [4, 8]
    //
    // coefficient = alpha / (1-beta1) = 0.5
    // updates = coefficient * [
    //    new_first_m1 / new_weighted_inf_norm1,
    //    new_first_m2 / new_weighted_inf_norm2
    // ] = [0.1, 0.1]
    // w = [
    //    w1_old - updates_1,
    //    w2_old - updates_2
    // ] = [1.9, 3.9]
    //
    expectArraysClose(await x.data(), [1.9, 3.9]);

    cost.dispose();
    numTensors = tf.memory().numTensors;

    cost = optimizer.minimize(f, /* returnCost */ false);

    // gradient = [3.8, 7.8]
    // new_first_m = [
    //    beta1 * old_first_m_w1 + (1-beta1) * grad_w1,
    //    beta1 * old_first_m_w2 + (1-beta1) * grad_w2
    // ] = [
    //    0.8 * 0.8 + 0.2 * 3.8,
    //    0.8 * 1.6 + 0.2 * 7.8
    // ] = [1.4, 2.84]
    //
    // ut_0 = beta2 * old_weighted_inf_norm = [
    //    0.9 * 4,
    //    0.9 * 8
    // ] = [3.6, 7.2]
    // u1_1 = [
    //    abs(grad_w1),
    //    abs(grad_w2)
    // ] = [3.8, 7.8]
    // new_weighted_inf_norm = max(ut_0, ut_1) = [3.8, 7.8]
    //
    // alpha = 0.1 / (1 + 0.1 * 1) = 0.0909090909
    //
    // coefficient = alpha / (1 - beta1*beta1) = 0.25252525
    // updates = coefficient * [
    //    new_first_m1 / new_weighted_inf_norm1,
    //    new_first_m2 / new_weighted_inf_norm2
    // ] = [0.09303, 0.09194]
    // w = [
    //    w1_old - updates_1,
    //    w2_old - updates_2
    // ] = [1.80697, 3.8086]
    //
    expectArraysClose(await x.data(), [1.80697, 3.8086]);
    // There should be no new additional Tensors.
    expect(tf.memory().numTensors).toBe(numTensors);

    expect(cost).toBe(null);

    x.dispose();
    optimizer.dispose();

    // The only tensor remaining should be the argument to variable().
    expect(tf.memory().numTensors).toBe(1);
  });

  it('serialization round-trip', () => {
    const originalOpt = tf.train.adamax(0.1, 0.2, 0.3, 2e-8, 0.1);
    const reserialized = tf.AdamaxOptimizer.fromConfig(
        tf.AdamaxOptimizer, originalOpt.getConfig());
    expect(reserialized.getConfig()).toEqual(originalOpt.getConfig());
  });
});
