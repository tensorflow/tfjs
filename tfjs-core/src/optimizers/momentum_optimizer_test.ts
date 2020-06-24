/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

describeWithFlags('MomentumOptimizer', ALL_ENVS, () => {
  it('basic', async () => {
    const learningRate = .1;
    const momentum = .5;
    const optimizer = tf.train.momentum(learningRate, momentum);

    const x = tf.tensor1d([1, 2]).variable();

    const f: () => tf.Scalar = () => x.square().sum();

    let numTensors = tf.memory().numTensors;

    let cost = optimizer.minimize(f, /* returnCost */ true);

    // Cost & velocity should be the only additional arrays.
    expect(tf.memory().numTensors).toBe(numTensors + 2);

    // newAccumulation = momentum * accumulation + gradient
    // newVariable += -learningRate * newAccumulation + variable
    //
    // de/dx = [2, 4]
    // newAccumulation = [2, 4]
    // x = [.8, 1.6]
    expectArraysClose(await x.data(), [.8, 1.6]);

    cost.dispose();
    numTensors = tf.memory().numTensors;

    cost = optimizer.minimize(f, /* returnCost */ false);

    // de/dx = [1.6, 3.2]
    // accumulation = [2, 4]
    // newAccumulation = [2.6, 5.2]
    // x = [0.54, 1.08]
    expectArraysClose(await x.data(), [0.54, 1.08]);

    // There should be no new additional Tensors.
    expect(tf.memory().numTensors).toBe(numTensors);

    expect(cost).toBe(null);

    x.dispose();
    numTensors = tf.memory().numTensors;
    optimizer.dispose();

    // The optimizer.dispose() call should have disposed the m variable and the
    // momentum variable for x.
    expect(tf.memory().numTensors).toBe(numTensors - 2);
  });

  it('basic - with Nesterov', async () => {
    const learningRate = .1;
    const momentum = .5;
    const useNesterov = true;
    const optimizer = tf.train.momentum(learningRate, momentum, useNesterov);

    const x = tf.tensor1d([1, 2]).variable();

    const f: () => tf.Scalar = () => x.square().sum();

    let numTensors = tf.memory().numTensors;

    let cost = optimizer.minimize(f, /* returnCost */ true);

    // Cost and velocity should be the only additional arrays.
    expect(tf.memory().numTensors).toBe(numTensors + 2);

    // newAccumulation = momentum * accumulation + gradient
    // newVariable = -learningRate * (newAccumulation * momentum + gradient) +
    //                variable
    //
    // de/dx = [2, 4]
    // newAccumulation = [2, 4]
    // newVariable = -0.1 * ([2, 4] * 0.5 + [2, 4]) + [1, 2]
    // x = [.7, 1.4]
    expectArraysClose(await x.data(), [.7, 1.4]);

    cost.dispose();
    numTensors = tf.memory().numTensors;

    cost = optimizer.minimize(f, /* returnCost */ false);

    // de/dx = [1.4, 2.8]
    // accumulation = [2, 4]
    // newAccumulation = [0.5 * 2 + 1.4, 0.5 * 4 + 2.8] = [2.4, 4.8]
    // newVariable = -0.1 * ([2.4, 4.8] * 0.5 + [1.4, 2.8]) + [0.7, 1.4]
    // x = [0.44, 0.88]
    expectArraysClose(await x.data(), [0.44, 0.88]);

    // There should be no new additional Tensors.
    expect(tf.memory().numTensors).toBe(numTensors);

    expect(cost).toBe(null);

    x.dispose();
    numTensors = tf.memory().numTensors;
    optimizer.dispose();

    // The optimizer.dispose() call should have disposed the m variable and the
    // momentum variable for x.
    expect(tf.memory().numTensors).toBe(numTensors - 2);
  });

  it('Save, load weights and conntinue training', async () => {
    const learningRate = .1;
    const momentum = .5;
    const useNesterov = true;
    const optimizer1 = tf.train.momentum(learningRate, momentum, useNesterov);

    const x = tf.tensor1d([1, 2]).variable();
    const f: () => tf.Scalar = () => x.square().sum();

    let cost = optimizer1.minimize(f, /* returnCost */ true);

    // The iterations counter and the accumulation for the variable x.
    const optimizer2 = tf.train.momentum(learningRate, momentum, useNesterov);
    await optimizer2.setWeights(await optimizer1.getWeights());
    cost = optimizer2.minimize(f, /* returnCost */ true);
    expectArraysClose(await cost.data(), 2.45);
    expectArraysClose(await x.data(), [0.44, 0.88]);
    expect(optimizer2.iterations).toEqual(2);
  });

  it('serialization round-trip', () => {
    const originalOpt = tf.train.momentum(0.1, 0.2, true);
    const reserialized = tf.MomentumOptimizer.fromConfig(
        tf.MomentumOptimizer, originalOpt.getConfig());
    expect(reserialized.getConfig()).toEqual(originalOpt.getConfig());
  });
});
