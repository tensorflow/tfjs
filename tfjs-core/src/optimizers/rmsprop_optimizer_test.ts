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

describeWithFlags('RMSPropOptimizer', ALL_ENVS, () => {
  it('basic', async () => {
    const learningRate = 0.1;
    const moment = 0.1;
    const rho = 0.95;
    const optimizer = tf.train.rmsprop(learningRate, rho, moment);

    const x = tf.tensor1d([1, 2]).variable();

    const f = () => x.square().sum() as tf.Scalar;

    let numTensors = tf.memory().numTensors;

    let cost = optimizer.minimize(f, /* returnCost */ true);

    // Cost & 2 accumulators should be the only additional arrays.
    expect(tf.memory().numTensors).toBe(numTensors + 3);

    // epsilon = 1e-8
    // newAccumulatedMeanSquare =
    //          rho * accumulatedMeanSquare + (1 - rho) * grad ^ 2 = (0.2)
    // newAccumulatedMoments = momentum * accumulatedMoments +
    //          learning_rate * gradient / sqrt(newAccumulatedMeanSquare +
    //          epsilon) = 0.1 * 0 + ((0.1 * 2) / sqrt(0.2 + 1e-8)) = 0.44721
    // x -= learningRate * newAccumulatedMoments
    //
    // de/dx = [2, 4]
    // accumulatedMeanSquare = [0, 0]
    // newAccumulatedMeanSquare = [.2, .8]
    // accumulatedMoments = [0, 0]
    // newAccumulatedMoments = [0.44721, 0.44721]
    // x = [0.55279, 1.55279]
    expectArraysClose(await x.data(), [0.55279, 1.55279]);

    cost.dispose();
    numTensors = tf.memory().numTensors;

    cost = optimizer.minimize(f, /* returnCost */ false);

    // x = [0.55279, 1.55279]
    // de/dx = [1.10558, 3.10558]
    // accumulatedMeanSquare = [0.2, 0.8]
    // newAccumulatedMeanSquare = [0.25105125, 1.242231]
    // accumulatedMoments = [0.44721, 0.44721]
    // newAccumulatedMoments = [0.26534, 0.32336]
    // x = [0.28745, 1.22943]

    // TODO: Fix numerical precision.
    expectArraysClose(await x.data(), [0.28745, 1.222943], 1e-2);

    // There should be no new additional Tensors.
    expect(tf.memory().numTensors).toBe(numTensors);

    expect(cost).toBe(null);

    x.dispose();
    optimizer.dispose();
    // The only tensor remaining is the argument to variable().
    expect(tf.memory().numTensors).toBe(1);
  });

  it('gradient with centered momentum', async () => {
    const learningRate = 0.1;
    const moment = 0.1;
    const rho = 0.95;
    const eps = 1e-8;
    const optimizer = tf.train.rmsprop(learningRate, rho, moment, eps, true);

    const x = tf.tensor1d([1, 2]).variable();

    const f = () => x.square().sum() as tf.Scalar;

    let numTensors = tf.memory().numTensors;

    let cost = optimizer.minimize(f, /* returnCost */ true);

    // Cost & 3 accumulators should be the only additional arrays.
    expect(tf.memory().numTensors).toBe(numTensors + 4);

    // epsilon = 1e-8
    // newAccumulatedMeanSquare =
    //          rho * accumulatedMeanSquare + (1 - rho) * grad ^ 2 = [.2, .8]
    // newAccumulatedMeanGrad =
    //          rho * accumulatedMeanGrad + (1 - rho) * grad = [0.1, 0.2]
    // newAccumulatedMoments = momentum * accumulatedMoments +
    //          learning_rate * gradient / sqrt(newAccumulatedMeanSquare
    //            - newAccumulatedMeanGrad * 2 +
    //              epsilon) = 0.1 * 0 + ((0.1 * 2)
    //                / sqrt(0.2 - 0.01 + 1e-8)) = 0.458831
    // x -= learningRate * newAccumulatedMoments
    //
    // de/dx = [2, 4]
    // accumulatedMeanSquare = [0, 0]
    // newAccumulatedMeanSquare = [.2, .8]
    // newAccumulatedMeanGrad = [.1, .2]
    // accumulatedMoments = [0, 0]
    // newAccumulatedMoments = [0.45883, 0.458831]
    // x = [0.54117, 1.541169]
    expectArraysClose(await x.data(), [0.54117, 1.541169]);

    cost.dispose();
    numTensors = tf.memory().numTensors;

    cost = optimizer.minimize(f, /* returnCost */ false);

    // x = [0.54117, 1.541169]
    // de/dx = [1.08234, 3.082338]
    // accumulatedMeanSquare = [0.2, 0.8]
    // accumulatedMeanGrad = [.1, .2]
    // newAccumulatedMeanSquare = [0.248572, 1.235040]
    // newAccumulatedMeanGrad = [0.149117, 0.3441169]
    // accumulatedMoments = [0.45883, 0.458831]
    // newAccumulatedMoments = [0.273385, 0.3375766]
    // x = [0.267785, 1.2035924]

    // TODO: Fix numerical precision.
    expectArraysClose(await x.data(), [0.267785, 1.2035924], 1e-2);

    // There should be no new additional Tensors.
    expect(tf.memory().numTensors).toBe(numTensors);

    expect(cost).toBe(null);

    x.dispose();
    optimizer.dispose();
    // The only tensor remaining is the argument to variable().
    expect(tf.memory().numTensors).toBe(1);
  });

  it('Save and load weigths: centered = false', async () => {
    const learningRate = 0.1;
    const moment = 0.1;
    const rho = 0.95;
    const optimizer1 = tf.train.rmsprop(learningRate, rho, moment);

    const x = tf.tensor1d([1, 2]).variable();
    const f = () => x.square().sum() as tf.Scalar;

    let cost = optimizer1.minimize(f, /* returnCost */ true);
    expectArraysClose(await cost.data(), 5);
    expectArraysClose(await x.data(), [0.5527865, 1.5527864]);

    const weights = await optimizer1.getWeights();
    // An iteration variable and two optimizer state variables.
    expect(weights.length).toEqual(3);

    const optimizer2 = tf.train.rmsprop(learningRate, rho, moment);
    await optimizer2.setWeights(weights);

    cost = optimizer2.minimize(f, /* returnCost */ true);
    expectArraysClose(await cost.data(), 2.7167187);
    expectArraysClose(await x.data(), [0.2874418, 1.2294267]);
    expect(optimizer2.iterations).toEqual(2);
  });

  it('Save, load weigths and continue training: centered = true', async () => {
    const learningRate = 0.1;
    const moment = 0.1;
    const rho = 0.95;
    const epsilon: number = undefined;
    const centered = true;
    const optimizer1 =
        tf.train.rmsprop(learningRate, rho, moment, epsilon, centered);

    const x = tf.tensor1d([1, 2]).variable();
    const f = () => x.square().sum() as tf.Scalar;

    let cost = optimizer1.minimize(f, /* returnCost */ true);
    expectArraysClose(await cost.data(), 5);
    expectArraysClose(await x.data(), [0.5411684, 1.5411685]);

    const weights = await optimizer1.getWeights();
    // An iteration variable and three optimizer state variables.
    expect(weights.length).toEqual(4);

    const optimizer2 =
        tf.train.rmsprop(learningRate, rho, moment, epsilon, centered);
    await optimizer2.setWeights(weights);

    cost = optimizer2.minimize(f, /* returnCost */ true);
    expectArraysClose(await cost.data(), 2.668063);
    expectArraysClose(await x.data(), [0.2677834, 1.2035918]);
    expect(optimizer2.iterations).toEqual(2);

    const optimizer3 =
        tf.train.rmsprop(learningRate, rho, moment, epsilon, centered);
    await optimizer3.setWeights(await optimizer2.getWeights());
    cost = optimizer3.minimize(f, /* returnCost */ true);
    expectArraysClose(await cost.data(), 1.520341);
    expect(optimizer3.iterations).toEqual(3);
  });

  it('serialization round-trip', () => {
    const originalOpt = tf.train.rmsprop(0.1, 0.5, 0.1, 1e-7, true);
    const reserialized = tf.RMSPropOptimizer.fromConfig(
        tf.RMSPropOptimizer, originalOpt.getConfig());
    expect(reserialized.getConfig()).toEqual(originalOpt.getConfig());
  });
});
