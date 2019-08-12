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

describeWithFlags('AdagradOptimizer', ALL_ENVS, () => {
  it('basic', async () => {
    const learningRate = .1;
    const initialAccumulatorValue = .1;
    const optimizer = tf.train.adagrad(learningRate, initialAccumulatorValue);

    const x = tf.tensor1d([1, 2]).variable();

    const f = () => x.square().sum() as tf.Scalar;

    let numTensors = tf.memory().numTensors;

    let cost = optimizer.minimize(f, /* returnCost */ true);

    // Cost & accumulator should be the only additional arrays.
    expect(tf.memory().numTensors).toBe(numTensors + 2);

    // epsilon = 1-e8
    // newAccumulatedGrad = accumulatedGrad + grad^2
    // x -= (learningRate * grad) / sqrt(newAccumulatedGrad + eps)
    // de/dx = [2, 4]
    // accumulatedGrad = [0.1, 0.1]
    // newAccumulatedGrad = [4.1, 16.1]
    // x = [0.9012270405, 1.900311042]
    expectArraysClose(await x.data(), [0.9012270405, 1.9003110428]);

    cost.dispose();
    numTensors = tf.memory().numTensors;

    cost = optimizer.minimize(f, /* returnCost */ false);

    // de/dx = [1.802454081, 3.9501555214]
    // accumulatedGrad = [4.1, 16.1]
    // newAccumulatedGrad = [7.3488407141, 31.7037286432]
    // x = [0.8347372764, 1.83015597828]

    // TODO: Fix numerical precision.
    expectArraysClose(await x.data(), [0.8347372764, 1.83015597828], 1e-2);

    // There should be no new additional Tensors.
    expect(tf.memory().numTensors).toBe(numTensors);

    expect(cost).toBe(null);

    x.dispose();
    optimizer.dispose();

    // The only tensor remaining is the argument to variable().
    expect(tf.memory().numTensors).toBe(1);
  });

  it('Continue training after loading weights', async () => {
    const learningRate = .1;
    const initialAccumulatorValue = .1;
    const optimizer1 = tf.train.adagrad(learningRate, initialAccumulatorValue);

    const x = tf.tensor1d([2, 4]).variable();
    const f = () => x.square().sum() as tf.Scalar;
    let cost = optimizer1.minimize(f, /* returnCost */ true);
    expectArraysClose(await cost.data(), 20);

    const weights = await optimizer1.getWeights();
    expect(weights.length).toEqual(2);
    expect(weights[0].name).toEqual('iter');
    expect(weights[1].name).toEqual(`${x.name}/accumulator`);

    const optimizer2 = tf.train.adam(learningRate, initialAccumulatorValue);
    await optimizer2.setWeights(weights);

    cost = optimizer2.minimize(f, /* returnCost */ true);
    expectArraysClose(await cost.data(), 18.82179);
    expect(optimizer2.iterations).toEqual(2);
  });

  it('serialization round-trip', () => {
    const originalOpt = tf.train.adagrad(0.1, 0.2);
    const reserialized = tf.AdagradOptimizer.fromConfig(
        tf.AdagradOptimizer, originalOpt.getConfig());
    expect(reserialized.getConfig()).toEqual(originalOpt.getConfig());
  });
});
