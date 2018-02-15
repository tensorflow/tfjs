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
import {AdamaxOptimizer} from './adamax_optimizer';

describeWithFlags('AdamaxOptimizer', ALL_ENVS, () => {
  it('basic', () => {
    const learningRate = 0.1;
    const beta1 = 0.8;
    const beta2 = 0.9;
    const decay = 0.1;
    const optimizer =
        dl.train.adamax(learningRate, beta1, beta2, undefined, decay);

    const x = dl.variable(dl.tensor1d([2, 4]));

    const f = () => x.square().sum() as dl.Scalar;

    let numTensors = dl.memory().numTensors;

    let cost = optimizer.minimize(f, /* returnCost */ true);

    // Cost & 2 accumulators should be the only additional arrays.
    expect(dl.memory().numTensors).toBe(numTensors + 3);
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
    expectArraysClose(x, [1.9, 3.9]);

    cost.dispose();
    numTensors = dl.memory().numTensors;

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
    expectArraysClose(x, [1.80697, 3.8086]);
    // There should be no new additional Tensors.
    expect(dl.memory().numTensors).toBe(numTensors);

    expect(cost).toBe(null);

    x.dispose();
    optimizer.dispose();

    // The only tensor remaining should be the argument to variable().
    expect(dl.memory().numTensors).toBe(1);
  });

  it('adamax', () => {
    const math = ENV.math;

    const inputProvider: InputProvider = {
      getNextCopy() {
        return Tensor1D.new([2, 4]);
      },
      disposeCopy(example) {}
    };

    dl.tidy(() => {
      const learningRate = 0.1;
      const beta1 = 0.8;
      const beta2 = 0.9;
      const decay = 0.1;
      const g = new Graph();
      const x = g.placeholder('x', [2]);
      const w = g.variable('w', dl.zeros([1, 2]));
      const b = g.variable('b', dl.zeros([1]));
      const y = g.reduceSum(g.add(g.matmul(w, x), b));
      const optimizer =
          new AdamaxOptimizer(learningRate, beta1, beta2, undefined, decay);
      const session = new Session(g, math);
      // w = reduce_sum(w_1*x_1 + w_2*x_2 + b)
      // new_first_m = [beta1*old_first_m_w1 + (1-beta1)*grad_w1,
      //                beta1*old_first_m_w2 + (1-beta1)*grad_w2]
      //             = [.4, .8]
      //
      // ut_0 = beta2*old_weighted_inf_norm = [0, 0]
      // u1_1 = [(1-beta2)*grad_w1, (1-beta2)*grad_w2] = [.2 .4]
      // new_weighted_inf_norm = max(ut_0, ut_1 ) = [.2 .4]
      //
      // coefficient = alpha/(1-beta1) = 0.5
      // updates = coefficient*[new_first_m1/new_weighted_inf_norm1,
      //                        new_first_m2/new_weighted_inf_norm2]
      //         = [1.0, 1.0]
      // w = [ w1_old - lr*updates_1, w2_old - lr*updates_2]
      //            = [-0.1, -0.1]
      //
      session.train(y, [{tensor: x, data: inputProvider}], 1, optimizer);
      const dydw = session.activationArrayMap.get(w).dataSync();
      expectArraysClose(dydw, new Float32Array([-0.1, -0.1]), 1e-1);

      // w = reduce_sum(w_1*x_1 + w_2*x_2 + b)
      // new_first_m = [beta1*old_first_m_w1 + (1-beta1)*grad_w1,
      //                beta1*old_first_m_w2 + (1-beta1)*grad_w2]
      //             = [0.8*0.4 + 0.2*2, 0.8*0.8 + 0.2*4]
      //             = [0.72, 1.44]
      //
      // ut_0 = beta2*old_weighted_inf_norm = [.18 .36]
      // u1_1 = [(1-beta2)*grad_w1, (1-beta2)*grad_w2] = [.2 .4]
      // new_weighted_inf_norm = max(ut_0, ut_1 ) = [.2 .4]
      //
      // alpha = 0.1 / (1 + 0.1 * 1) = 0.09090909
      //
      // coefficient = alpha/(1-(beta1*beta1) = 0.25252525
      // updates = coefficient*[new_first_m1/new_weighted_inf_norm1,
      //                        new_first_m2/new_weighted_inf_norm2]
      //         = [0.909090909, 0.909090909]
      // w = [ w1_old - lr*updates_1, w2_old - lr*updates_2]
      //            = [-0.1909090909, -0.1909090909]

      session.train(y, [{tensor: x, data: inputProvider}], 1, optimizer);
      const dydw2 = session.activationArrayMap.get(w).dataSync();
      expectArraysClose(
          dydw2, new Float32Array([-0.1909090909, -0.1909090909]), 1e-2);
    });
  });
});
