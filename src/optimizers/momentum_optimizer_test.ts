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
import {MomentumOptimizer} from './momentum_optimizer';

describeWithFlags('MomentumOptimizer', ALL_ENVS, () => {
  it('basic', () => {
    const learningRate = .1;
    const momentum = .5;
    const optimizer = dl.train.momentum(learningRate, momentum);

    const x = dl.tensor1d([1, 2]).variable();

    const f = () => x.square().sum() as dl.Scalar;

    let numTensors = dl.memory().numTensors;

    let cost = optimizer.minimize(f, /* returnCost */ true);

    // Cost / velocity should be the only additional arrays.
    expect(dl.memory().numTensors).toBe(numTensors + 2);

    // newAccumulation = momentum * accumulation + gradient
    // newVariable += -learningRate * newAccumulation + variable
    //
    // de/dx = [2, 4]
    // newAccumulation = [2, 4]
    // x = [.8, 1.6]
    expectArraysClose(x, [.8, 1.6]);

    cost.dispose();
    numTensors = dl.memory().numTensors;

    cost = optimizer.minimize(f, /* returnCost */ false);

    // de/dx = [1.6, 3.2]
    // accumulation = [2, 4]
    // newAccumulation = [2.6, 5.2]
    // x = [0.54, 1.08]
    expectArraysClose(x, [0.54, 1.08]);

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
      disposeCopy(example) {}
    };

    dl.tidy(() => {
      const g = new Graph();
      const x = g.placeholder('x', [2]);
      const w = g.variable('w', dl.zeros([1, 2]));
      const b = g.variable('b', dl.zeros([1]));
      const y = g.reduceSum(g.add(g.matmul(w, x), b));
      const optimizer = new MomentumOptimizer(0.1, 0.5);
      const session = new Session(g, math);
      // w = reduce_sum(w_1*x_1 + w_2*x_2 + b)
      // velocity_w = [momentum* old_vel_w1 + x_1,
      //                momentum* old_vel_w2 + x_2] = [2,4]
      // w = [ w_old - lr*vel_w1, w_old - lr*vel_w2] = [-0.2, -0.4]
      session.train(y, [{tensor: x, data: inputProvider}], 1, optimizer);
      let wValue = session.activationArrayMap.get(w).dataSync();
      expectArraysClose(wValue, new Float32Array([-.2, -0.4]));
      // velocity_w = [momentum* old_vel_w1 + x_1,
      //                momentum* old_vel_w2 + x_2] = [3,6]
      // w = [ w_old - lr*vel_w1, w_old - lr*vel_w2] = [-0.5, -1.0]
      session.train(y, [{tensor: x, data: inputProvider}], 1, optimizer);
      wValue = session.activationArrayMap.get(w).dataSync();
      expectArraysClose(wValue, new Float32Array([-.5, -1.0]));
    });
  });

  it('basic - with Nesterov', () => {
    const learningRate = .1;
    const momentum = .5;
    const useNesterov = true;
    const optimizer = dl.train.momentum(learningRate, momentum, useNesterov);

    const x = dl.tensor1d([1, 2]).variable();

    const f = () => x.square().sum() as dl.Scalar;

    let numTensors = dl.memory().numTensors;

    let cost = optimizer.minimize(f, /* returnCost */ true);

    // Cost / velocity should be the only additional arrays.
    expect(dl.memory().numTensors).toBe(numTensors + 2);

    // newAccumulation = momentum * accumulation + gradient
    // newVariable = -learningRate * (newAccumulation * momentum + gradient) +
    //                variable
    //
    // de/dx = [2, 4]
    // newAccumulation = [2, 4]
    // newVariable = -0.1 * ([2, 4] * 0.5 + [2, 4]) + [1, 2]
    // x = [.7, 1.4]
    expectArraysClose(x, [.7, 1.4]);

    cost.dispose();
    numTensors = dl.memory().numTensors;

    cost = optimizer.minimize(f, /* returnCost */ false);

    // de/dx = [1.4, 2.8]
    // accumulation = [2, 4]
    // newAccumulation = [0.5 * 2 + 1.4, 0.5 * 4 + 2.8] = [2.4, 4.8]
    // newVariable = -0.1 * ([2.4, 4.8] * 0.5 + [1.4, 2.8]) + [0.7, 1.4]
    // x = [0.44, 0.88]
    expectArraysClose(x, [0.44, 0.88]);

    // There should be no new additional Tensors.
    expect(dl.memory().numTensors).toBe(numTensors);

    expect(cost).toBe(null);

    x.dispose();
    optimizer.dispose();

    // The only tensor remaining is the argument to variable().
    expect(dl.memory().numTensors).toBe(1);
  });

  it('graph - with Nesterov', () => {
    const math = ENV.math;

    const inputProvider: InputProvider = {
      getNextCopy() {
        return Tensor1D.new([2, 4]);
      },
      disposeCopy(example) {}
    };

    dl.tidy(() => {
      const g = new Graph();
      const x = g.placeholder('x', [2]);
      const w = g.variable('w', dl.zeros([1, 2]));
      const b = g.variable('b', dl.zeros([1]));
      const y = g.reduceSum(g.add(g.matmul(w, x), b));
      const optimizer = new MomentumOptimizer(0.1, 0.5, undefined, true);
      const session = new Session(g, math);
      // w = reduce_sum(w_1 * x_1 + w_2 * x_2 + b) = [0, 0]
      // velocity_w = [momentum * old_vel_w1 + x_1,
      //                momentum * old_vel_w2 + x_2] = [2, 4]
      // w = [
      //     w1_old - lr * (vel_w1 * momentum + x_1),
      //     w2_old - lr * (vel_w2 * momentum + x_2)
      // ] = [
      //     0 - 0.1 * (2 * 0.5 + 2),
      //     0 - 0.1 * (4 * 0.5 + 4),
      // ]
      session.train(y, [{tensor: x, data: inputProvider}], 1, optimizer);
      let wValue = session.activationArrayMap.get(w).dataSync();
      expectArraysClose(wValue, new Float32Array([-.3, -0.6]));
      // w = reduce_sum(w_1 * x_1 + w_2 * x_2 + b) = [-0.3, -0.6]
      // velocity_w = [
      //    momentum * old_vel_w1 + x_1,
      //    momentum * old_vel_w2 + x_2
      // ] = [
      //    0.5 * 2 + 2,
      //    0.5 * 4 + 4
      // ] = [3, 6]
      // w = [
      //     w1_old - lr * (vel_w1 * momentum + x_1),
      //     w2_old - lr * (vel_w2 * momentum + x_2)
      // ] = [
      //     -0.3 - 0.1 * (3 * 0.5 + 2),
      //     -0.6 - 0.1 * (6 * 0.5 + 4),
      // ] = [-0.65, -1.3]
      session.train(y, [{tensor: x, data: inputProvider}], 1, optimizer);
      wValue = session.activationArrayMap.get(w).dataSync();
      expectArraysClose(wValue, new Float32Array([-.65, -1.3]));
    });
  });
});
