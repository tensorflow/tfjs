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

describeWithFlags('SGDOptimizer', ALL_ENVS, () => {
  it('basic', async () => {
    const learningRate = .1;
    const optimizer = tf.train.sgd(learningRate);

    const x = tf.scalar(4).variable();

    let numTensors = tf.memory().numTensors;

    let cost = optimizer.minimize(() => x.square(), /* returnCost */ true);

    // Cost should be the only additional arrays.
    expect(tf.memory().numTensors).toBe(numTensors + 1);

    // de/dx = 2x
    const expectedValue1 = -2 * 4 * learningRate + 4;
    expectArraysClose(await x.data(), [expectedValue1]);
    expectArraysClose(await cost.data(), [Math.pow(4, 2)]);

    cost.dispose();
    numTensors = tf.memory().numTensors;

    cost = optimizer.minimize(() => x.square(), /* returnCost */ false);
    // There should be no new additional Tensors.
    expect(tf.memory().numTensors).toBe(numTensors);

    const expectedValue2 = -2 * expectedValue1 * learningRate + expectedValue1;
    expectArraysClose(await x.data(), [expectedValue2]);
    expect(cost).toBe(null);

    optimizer.dispose();
    x.dispose();
    // The only tensor remaining is the argument to variable().
    expect(tf.memory().numTensors).toBe(1);
  });

  it('Set and get weights: empty', async () => {
    const x = tf.scalar(4).variable();

    const learningRate = .1;
    const optimizer1 = tf.train.sgd(learningRate);

    let weights = await optimizer1.getWeights();
    expect(optimizer1.iterations).toEqual(0);

    optimizer1.minimize(() => x.square());

    weights = await optimizer1.getWeights();
    expect(optimizer1.iterations).toEqual(1);
    expect(weights.length).toEqual(1);
    expect(weights[0].name).toEqual('iter');
    expectArraysClose(await weights[0].tensor.data(), 1);

    const optimizer2 = tf.train.sgd(learningRate);
    await optimizer2.setWeights(weights);
    optimizer2.minimize(() => x.square());
    expectArraysClose(await x.data(), 2.56);
    expect(optimizer2.iterations).toEqual(2);

    const optimizer3 = tf.train.sgd(learningRate);
    await optimizer3.setWeights(await optimizer2.getWeights());
    optimizer3.minimize(() => x.square());
    expectArraysClose(await x.data(), 2.048);
    expect(optimizer3.iterations).toEqual(3);
  });

  it('serialization round-trip', () => {
    const learningRate = .1;
    const originalOpt = tf.train.sgd(learningRate);
    const reserialized =
        tf.SGDOptimizer.fromConfig(tf.SGDOptimizer, originalOpt.getConfig());
    expect(reserialized.getConfig()).toEqual(originalOpt.getConfig());
  });
});
