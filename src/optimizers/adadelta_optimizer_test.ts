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
import {ALL_ENVS, expectArraysClose} from '../test_util';
import {describeWithFlags} from '../jasmine_util';

describeWithFlags('AdadeltaOptimizer', ALL_ENVS, () => {
  it('basic', () => {
    const learningRate = .1;
    const rho = .95;
    const optimizer = tf.train.adadelta(learningRate, rho);

    const x = tf.tensor1d([1, 2]).variable();

    const f = () => x.square().sum() as tf.Scalar;

    let numTensors = tf.memory().numTensors;

    let cost = optimizer.minimize(f, /* returnCost */ true);

    // Cost & 2 accumulators should be the only additional arrays.
    expect(tf.memory().numTensors).toBe(numTensors + 3);

    // epsilon = 1-e8
    // newAccumulatedGrad = rho * accumulatedGrad + (1 - rho) * grad ^ 2
    // updates = -grad * sqrt(accumulatedUpdate + epsilon) /
    //     sqrt(accumulatedGrad + epsilon)
    // newAccumulatedUpdate = rho * accumulatedUpdate + (1 - rho) * updates ^ 2
    // x += learningRate * updates
    //
    // de/dx = [2, 4]
    // accumulatedGrad = [0, 0]
    // newAccumulatedGrad = [.2, .8]
    // updates = [-2, -4]
    // newAccumulatedUpdate = [.2, .8]
    // x = [0.8, 1.6]
    expectArraysClose(x, [0.8, 1.6]);

    cost.dispose();
    numTensors = tf.memory().numTensors;

    cost = optimizer.minimize(f, /* returnCost */ false);

    // de/dx = [1.6, 3.2]
    // accumulatedGrad = [.2, .8]
    // accumulatedUpdate = [.2, .8]
    // newAccumulatedGrad = [0.318, 1.272]
    // updates = [-1.6, -3.2]
    // x = [0.64, 1.28]
    expectArraysClose(x, [0.64, 1.28]);

    // There should be no new additional Tensors.
    expect(tf.memory().numTensors).toBe(numTensors);

    expect(cost).toBe(null);

    x.dispose();
    optimizer.dispose();

    // The only tensor remaining is the argument to variable().
    expect(tf.memory().numTensors).toBe(1);
  });
});
