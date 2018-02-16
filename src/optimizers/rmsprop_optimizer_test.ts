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
import {InputProvider} from '../data/input_provider';
import {ENV} from '../environment';
import {Graph} from '../graph/graph';
import {Session} from '../graph/session';
import * as dl from '../index';
import {ALL_ENVS, describeWithFlags, expectArraysClose} from '../test_util';

import {RMSPropOptimizer} from './rmsprop_optimizer';

describeWithFlags('RMSPropOptimizer', ALL_ENVS, () => {
  it('basic', () => {
    const learningRate = 0.1;
    const moment = 0.1;
    const rho = 0.95;
    const optimizer = dl.train.rmsprop(learningRate, rho, moment);

    const x = dl.tensor1d([1, 2]).variable();

    const f = () => x.square().sum() as dl.Scalar;

    let numTensors = dl.memory().numTensors;

    let cost = optimizer.minimize(f, /* returnCost */ true);

    // Cost & 2 accumulators should be the only additional arrays.
    expect(dl.memory().numTensors).toBe(numTensors + 3);

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
    expectArraysClose(x, [0.55279, 1.55279]);

    cost.dispose();
    numTensors = dl.memory().numTensors;

    cost = optimizer.minimize(f, /* returnCost */ false);

    // x = [0.55279, 1.55279]
    // de/dx = [1.10558, 3.10558]
    // accumulatedMeanSquare = [0.2, 0.8]
    // newAccumulatedMeanSquare = [0.25105125, 1.242231]
    // accumulatedMoments = [0.44721, 0.44721]
    // newAccumulatedMoments = [0.26534, 0.32336]
    // x = [0.28745, 1.22943]

    // TODO: Fix numerical precision.
    expectArraysClose(x, [0.28745, 1.222943], 1e-2);

    // There should be no new additional Tensors.
    expect(dl.memory().numTensors).toBe(numTensors);

    expect(cost).toBe(null);

    x.dispose();
    optimizer.dispose();
    // The only tensor remaining is the argument to variable().
    expect(dl.memory().numTensors).toBe(1);
  });

  it('graph', () => {
    const math = ENV.math;

    const inputProvider: InputProvider = {
      getNextCopy() {
        return dl.tensor1d([2, 4]);
      },
      disposeCopy(math) {}
    };

    dl.tidy(() => {
      const g = new Graph();
      const learningRate = 0.1;
      const moment = 0.1;
      const rho = 0.95;
      const x = g.placeholder('x', [2]);
      const w = g.variable('w', dl.zeros([1, 2]));
      const b = g.variable('b', dl.zeros([1]));
      const y = g.reduceSum(g.add(g.matmul(w, x), b));

      const session = new Session(g, math);
      const optimizer = new RMSPropOptimizer(learningRate, rho, moment);
      session.train(y, [{tensor: x, data: inputProvider}], 1, optimizer);
      const dydw = session.activationArrayMap.get(w).dataSync();
      // w = reduce_sum(w_1*x_1 + w_2*x_2 + b)
      // mean_sq = [
      //    rho * old_mean_sq_w1 + (1-rho) * grad_w1**2,
      //    rho * old_mean_sq_w2 + (1-rho) * grad_w2**2
      // ] = [0.2, 0.8]
      // moment = [
      //    momentum * old_moment_w1 + (lr * grad_w1) / sqrt(mean_sq_w1 + eps),
      //    momentum * old_moment_w2 + (lr * grad_w2) / sqrt(mean_sq_w2 + eps)
      // ] = [
      //    0.1 * 0 + (0.1 * 2) / sqrt(0.2 + 1e-8),
      //    0.1 * 0 + (0.1 * 4) / sqrt(0.8 + 1e-8)
      // ] = [0.44721358, 0.4472136]
      // w = [
      //    w1_old - moment_w1,
      //    w2_old - moment_w2
      // ] = [-0.44721358, -0.4472136]
      expectArraysClose(dydw, new Float32Array([-0.44721358, -0.4472136]));

      session.train(y, [{tensor: x, data: inputProvider}], 1, optimizer);
      const dydw2 = session.activationArrayMap.get(w).dataSync();
      // mean_sq = [
      //    rho * old_mean_sq_w1 + (1-rho) * grad_w1**2,
      //    rho * old_mean_sq_w2 + (1-rho) * grad_w2**2
      // ] = [
      //    0.95 * 0.2 + (1 - 0.95) * 4,
      //    0.95 * 0.8 + (1 - 0.95) * 8
      // ] = [0.39, 1.16]
      // moment = [
      //    momentum * old_moment_w1 + (lr * grad_w1) / sqrt(mean_sq_w1 + eps),
      //    momentum * old_moment_w2 + (lr * grad_w2) / sqrt(mean_sq_w2 + eps)
      // ] = [
      //    0.1 * 0.44721358 + (0.1 * 2) / sqrt(0.39 + 1e-8),
      //    0.1 * 0.4472136  + (0.1 * 4) / sqrt(1.16 + 1e-8)
      // ] = [0.364977, 0.3649]
      // w = [
      //    w1_old - moment_w1,
      //    w2_old - moment_w2
      // ] = [-0.812191, -0.812191]
      expectArraysClose(dydw2, new Float32Array([-0.812191, -0.812191]));
    });
  });
});
