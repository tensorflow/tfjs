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
import {Tensor1D} from '../tensor';
import {ALL_ENVS, describeWithFlags, expectArraysClose} from '../test_util';
import {AdadeltaOptimizer} from './adadelta_optimizer';

describeWithFlags('AdadeltaOptimizer', ALL_ENVS, () => {
  it('basic', () => {
    const learningRate = .1;
    const rho = .95;
    const optimizer = dl.train.adadelta(learningRate, rho);

    const x = dl.variable(dl.tensor1d([1, 2]));

    const f = () => x.square().sum() as dl.Scalar;

    let numTensors = dl.memory().numTensors;

    let cost = optimizer.minimize(f, /* returnCost */ true);

    // Cost & 2 accumulators should be the only additional arrays.
    expect(dl.memory().numTensors).toBe(numTensors + 3);

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
    numTensors = dl.memory().numTensors;

    cost = optimizer.minimize(f, /* returnCost */ false);

    // de/dx = [1.6, 3.2]
    // accumulatedGrad = [.2, .8]
    // accumulatedUpdate = [.2, .8]
    // newAccumulatedGrad = [0.318, 1.272]
    // updates = [-1.6, -3.2]
    // x = [0.64, 1.28]
    expectArraysClose(x, [0.64, 1.28]);

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
        return Tensor1D.new([2, 4]);
      },
      disposeCopy(math) {}
    };

    dl.tidy(() => {
      const g = new Graph();
      const x = g.placeholder('x', [2]);
      const w = g.variable('w', dl.zeros([1, 2]));
      const b = g.variable('b', dl.zeros([1]));
      const y = g.reduceSum(g.add(g.matmul(w, x), b));
      const session = new Session(g, math);
      const optimizer = new AdadeltaOptimizer(0.1, 0.8);
      // w = reduce_sum(w_1*x_1 + w_2*x_2 + b)
      // cache = [gamma*old_cache_w1 + (1-gamma)*grad_w1**2,
      //            gamma*old_cache_w2 + (1-gamma)*grad_w2**2]
      //            = [.8, 3.2]
      // updates = [sqrt(old_updates_w1 + eps)/sqrt(old_cache_w1 +
      // eps)*grad_w1,
      //            sqrt(old_updates_w2 + eps)/sqrT(old_cache_w2 +
      //            eps)*grad_w2] = [2, 4]
      // w = [ w1_old - lr*updates_w1,
      //            w2_old - lr*updates_w2]
      //            = [-0.2, -0.4]
      // new_updates = [gamma * old_updates_w1 + (1 - gamma) * 2**2,
      //                gamma * old_updates_w2 + (1 - gamma) * 4**2]
      //             = [0.8, 3.2]
      //
      session.train(y, [{tensor: x, data: inputProvider}], 1, optimizer);
      const dydw = session.activationArrayMap.get(w).dataSync();
      expectArraysClose(dydw, new Float32Array([-0.2, -0.4]), 1e-2);
      // cache = [gamma*old_cache_w1 + (1-gamma)*grad_w1**2,
      //            gamma*old_cache_w2 + (1-gamma)*grad_w2**2]
      //            = [1.44, 5.76]
      // updates = [sqrt(old_updates_w1 + eps)/sqrt(old_cache_w1 +
      // eps)*grad_w1,
      //            sqrt(old_updates_w2 + eps)/sqrT(old_cache_w2 +
      //            eps)*grad_w2] = [2, 4]
      // w = [ w1_old - lr*updates_w1,
      //            w2_old - lr*updates_w2]
      //            = [-0.4, -0.8]
      session.train(y, [{tensor: x, data: inputProvider}], 1, optimizer);
      const dydw2 = session.activationArrayMap.get(w).dataSync();
      expectArraysClose(dydw2, new Float32Array([-.4, -.8]), 1e-2);
    });
  });
});
