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
import {AdamOptimizer} from './adam_optimizer';

describeWithFlags('AdamOptimizer', ALL_ENVS, () => {
  it('basic', () => {
    const learningRate = .1;
    const beta1 = .8;
    const beta2 = .9;
    const optimizer = dl.train.adam(learningRate, beta1, beta2);

    const x = dl.tensor1d([2, 4]).variable();

    const f = () => x.square().sum() as dl.Scalar;

    let numTensors = dl.memory().numTensors;

    let cost = optimizer.minimize(f, /* returnCost */ true);

    // Cost & 2 accumulators should be the only additional arrays.
    expect(dl.memory().numTensors).toBe(numTensors + 3);
    // new_first_m = [
    //    beta1 * old_first_m_w1 + (1-beta1) * grad_w1,
    //    beta1 * old_first_m_w2 + (1-beta1) * grad_w2
    // ] = [.8, 1.6]
    // new_second_m = [
    //    beta2 * old_second_m_w1 + (1-beta2) * grad_w1**2,
    //    beta2 * old_second_m_w2 + (1-beta2) * grad_w2**2
    // ] = [1.6, 6.4]
    // m = [new_first_m/(1-acc_beta1)] = [4, 8]
    // v = [new_second_m/(1-acc_beta2)] = [16, 64]
    // x = [x - lr * m / sqrt(v)] = [1.9, 3.9]
    //
    expectArraysClose(x, [1.9, 3.9]);

    cost.dispose();
    numTensors = dl.memory().numTensors;

    cost = optimizer.minimize(f, /* returnCost */ false);

    // new_first_m = [
    //    beta1 * old_first_m_w1 + (1-beta1) * grad_w1,
    //    beta1 * old_first_m_w2 + (1-beta1) * grad_w2
    // ] = [1.4, 2.84]
    // new_second_m = [
    //    beta2 * old_second_m_w1 + (1-beta2) * grad_w1**2,
    //    beta2 * old_second_m_w2 + (1-beta2) * grad_w2**2
    // ] = [2.884, 11.884]
    // m = [new_first_m/(1-acc_beta1)] = [3.888888, 7.88889]
    // v = [new_second_m/(1-acc_beta2)] = [15.1789, 62.5473]
    // x = [x - lr * m / sqrt(v)] = [1.8000001, 3.8002]
    //
    expectArraysClose(x, [1.8000001, 3.8002]);
    // There should be no new additional Tensors.
    expect(dl.memory().numTensors).toBe(numTensors);

    expect(cost).toBe(null);

    x.dispose();
    optimizer.dispose();

    // The only tensor remaining should be the argument to variable().
    expect(dl.memory().numTensors).toBe(1);
  });

  it('graph', () => {
    const math = ENV.math;

    const inputProvider: InputProvider = {
      getNextCopy() {
        return dl.tensor1d([2, 4]);
      },
      disposeCopy(example) {}
    };

    dl.tidy(() => {
      const learningRate = .1;
      const beta1 = .8;
      const beta2 = .9;

      const g = new Graph();
      const x = g.placeholder('x', [2]);
      const w = g.variable('w', dl.zeros([1, 2]));
      const b = g.variable('b', dl.zeros([1]));
      const y = g.reduceSum(g.add(g.matmul(w, x), b));
      const optimizer = new AdamOptimizer(learningRate, beta1, beta2);
      const session = new Session(g, math);
      // w = reduce_sum(w_1*x_1 + w_2*x_2 + b)
      // new_first_m = [beta1*old_first_m_w1 + (1-beta1)*grad_w1,
      //                beta1*old_first_m_w2 + (1-beta1)*grad_w2]
      //             = [.4, .8]
      // new_second_m = [beta2*old_second_m_w1 + (1-beta2)*grad_w1**2,
      //                 beta2*old_second_m_w2 + (1-beta2)*grad_w2**2]
      //              = [.4, 1.6]
      // m = [new_first_m/(1-acc_beta1)] = [2, 4]
      // v = [new_second_m/(1-acc_beta2)] = [4, 16]
      // updates = [m_1/(sqrt(v_1) + eps),
      //            m_2/(sqrt(v_2) + eps)]
      //            = [1.0, 1.0]
      // w = [ w1_old - lr*updates_1, w2_old - lr*updates_2]
      //            = [-0.1, -0.1]
      //
      session.train(y, [{tensor: x, data: inputProvider}], 1, optimizer);
      const dydw = session.activationArrayMap.get(w).dataSync();
      expectArraysClose(dydw, new Float32Array([-0.1, -0.1]));
      // new_first_m = [beta1*old_first_m_w1 + (1-beta1)*grad_w1,
      //                beta1*old_first_m_w2 + (1-beta1)*grad_w2]
      //             = [0.8*0.4 + 0.2*2, 0.8*0.8 + 0.2*4]
      //             = [0.72, 1.44]
      // new_second_m = [beta2*old_second_m_w1 + (1-beta2)*grad_w1**2,
      //                 beta2*old_second_m_w2 + (1-beta2)*grad_w2**2]
      //              = [0.9*0.4 + 0.1*4, 0.9*1.6+0.1*16]
      //              = [0.76, 3.04]
      // m = [new_first_m/(1-acc_beta1)] = [2, 4]
      // v = [new_second_m/(1-acc_beta2)] = [4, 16]
      // updates = [m_1/sqrt(v_1) + eps,
      //            m_2/sqrt(v_2) + eps]
      //            = [1.0, 1.0]
      // w = [ w1_old - lr*updates_1, w2_old - lr*updates_2]
      //            = [-0.2, -0.2]
      session.train(y, [{tensor: x, data: inputProvider}], 1, optimizer);
      const dydw2 = session.activationArrayMap.get(w).dataSync();
      expectArraysClose(dydw2, new Float32Array([-.2, -.2]));
    });
  });
});
