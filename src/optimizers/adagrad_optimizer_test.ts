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
import {AdagradOptimizer} from './adagrad_optimizer';

describeWithFlags('AdagradOptimizer', ALL_ENVS, () => {
  it('basic', () => {
    const learningRate = .1;
    const initialAccumulatorValue = .1;
    const optimizer = dl.train.adagrad(learningRate, initialAccumulatorValue);

    const x = dl.variable(dl.tensor1d([1, 2]));

    const f = () => x.square().sum() as dl.Scalar;

    let numTensors = dl.memory().numTensors;

    let cost = optimizer.minimize(f, /* returnCost */ true);

    // Cost & accumulator should be the only additional arrays.
    expect(dl.memory().numTensors).toBe(numTensors + 2);

    // epsilon = 1-e8
    // newAccumulatedGrad = accumulatedGrad + grad^2
    // x -= (learningRate * grad) / sqrt(newAccumulatedGrad + eps)
    // de/dx = [2, 4]
    // accumulatedGrad = [0.1, 0.1]
    // newAccumulatedGrad = [4.1, 16.1]
    // x = [0.9012270405, 1.900311042]
    expectArraysClose(x, [0.9012270405, 1.9003110428]);

    cost.dispose();
    numTensors = dl.memory().numTensors;

    cost = optimizer.minimize(f, /* returnCost */ false);

    // de/dx = [1.802454081, 3.9501555214]
    // accumulatedGrad = [4.1, 16.1]
    // newAccumulatedGrad = [7.3488407141, 31.7037286432]
    // x = [0.8347372764, 1.83015597828]

    // TODO: Fix numerical precision.
    expectArraysClose(x, [0.8347372764, 1.83015597828], 1e-2);

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
      const optimizer = new AdagradOptimizer(0.1);
      const session = new Session(g, math);
      // w = reduce_sum(w_1*x_1 + w_2*x_2 + b)
      // cache = [old_cache_w1 + grad_w1**2,
      //                old_cache_w2 + grad_w2**2] = [4,16]
      // w = [ w1_old - lr*grad_w1/sqrt(cahce_w2 + eps),
      //                w2_old - lr*grad_w1/sqrt(cahce_w2 + eps)]
      //                = [-0.1, -0.1]
      session.train(y, [{tensor: x, data: inputProvider}], 1, optimizer);
      const dydw = session.activationArrayMap.get(w).dataSync();
      expectArraysClose(dydw, new Float32Array([-.1, -0.1]));
      // cache = [old_cache_w1 + grad_w1**2,
      //                old_cache_w2 + grad_w2**2] = [4,16]
      // w = [ w1_old - lr*grad_w1/sqrt(cahce_w2 + eps),
      //                w2_old - lr*grad_w1/sqrt(cahce_w2 + eps)]
      //                = [-0.1707, -0.1707]
      session.train(y, [{tensor: x, data: inputProvider}], 1, optimizer);
      const dydw2 = session.activationArrayMap.get(w).dataSync();
      expectArraysClose(dydw2, new Float32Array([-.1707, -.1707]));
    });
  });
});
