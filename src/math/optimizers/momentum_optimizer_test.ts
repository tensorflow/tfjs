/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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
import {InputProvider} from '../../data/input_provider';
import {ENV} from '../../environment';
import {Graph} from '../../graph/graph';
import {Session} from '../../graph/session';
import * as dl from '../../index';
import {Scalar, Tensor1D, variable} from '../../math/tensor';
import * as test_util from '../../test_util';
import {MathTests} from '../../test_util';

import {MomentumOptimizer} from './momentum_optimizer';

const tests: MathTests = it => {
  it('basic', math => {
    const learningRate = .1;
    const momentum = .5;
    const optimizer = dl.train.momentum(learningRate, momentum);

    const w = variable(dl.zeros([1, 2]));
    const b = dl.zeros([1]);
    const x = dl.tensor1d([2, 4]);

    // TODO(nsthorat): Use tensordot() instead of reshapes when it's ready.
    const f = () =>
        w.reshape([1, 2]).matMul(x.reshape([2, 1])).add(b).sum() as Scalar;

    let numTensors = dl.memory().numTensors;

    let cost = optimizer.minimize(f, /* returnCost */ true);

    // Cost / velocity should be the only additional arrays.
    expect(dl.memory().numTensors).toBe(numTensors + 2);

    // w = reduce_sum(w_1*x_1 + w_2*x_2 + b)
    // velocity_w = [momentum* old_vel_w1 + x_1,
    //                momentum* old_vel_w2 + x_2] = [2,4]
    // w = [ w_old - lr*vel_w1, w_old - lr*vel_w2] = [-0.2, -0.4]
    test_util.expectArraysClose(w, [-0.2, -0.4]);

    cost.dispose();
    numTensors = dl.memory().numTensors;

    cost = optimizer.minimize(f, /* returnCost */ false);

    // velocity_w = [momentum* old_vel_w1 + x_1,
    //                momentum* old_vel_w2 + x_2] = [3,6]
    // w = [ w_old - lr*vel_w1, w_old - lr*vel_w2] = [-0.5, -1.0]
    test_util.expectArraysClose(w, new Float32Array([-.5, -1.0]));

    // There should be no new additional Tensors.
    expect(dl.memory().numTensors).toBe(numTensors);

    expect(cost).toBe(null);

    x.dispose();
    b.dispose();
    w.dispose();
    optimizer.dispose();

    // There should be no more Tensors.
    expect(dl.memory().numTensors).toBe(0);
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
      test_util.expectArraysClose(wValue, new Float32Array([-.2, -0.4]));
      // velocity_w = [momentum* old_vel_w1 + x_1,
      //                momentum* old_vel_w2 + x_2] = [3,6]
      // w = [ w_old - lr*vel_w1, w_old - lr*vel_w2] = [-0.5, -1.0]
      session.train(y, [{tensor: x, data: inputProvider}], 1, optimizer);
      wValue = session.activationArrayMap.get(w).dataSync();
      test_util.expectArraysClose(wValue, new Float32Array([-.5, -1.0]));
    });
  });
};

test_util.describeMathCPU('MomentumOptimizer', [tests]);
test_util.describeMathGPU('MomentumOptimizer', [tests], [
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
]);
