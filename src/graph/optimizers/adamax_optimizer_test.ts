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
import {Array1D, NDArray} from '../../math/ndarray';
import * as test_util from '../../test_util';
import {Graph} from '../graph';
import {Session} from '../session';

import {AdamaxOptimizer} from './adamax_optimizer';

describe('adamax optimizer', () => {
  it('adamax', () => {
    const math = ENV.math;

    const inputProvider: InputProvider = {
      getNextCopy() {
        return Array1D.new([2, 4]);
      },
      disposeCopy(math, example) {}
    };

    math.scope(() => {
      const g = new Graph();
      const x = g.placeholder('x', [2]);
      const w = g.variable('w', NDArray.zeros([1, 2]));
      const b = g.variable('b', NDArray.zeros([1]));
      const y = g.reduceSum(g.add(g.matmul(w, x), b));
      const optimizer = new AdamaxOptimizer(0.1, 0.8, 0.9);
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
      test_util.expectArraysClose(dydw, new Float32Array([-0.1, -0.1]), 1e-5);

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
      // coefficient = alpha/(1-(beta1*beta1) = 0.278
      // updates = coefficient*[new_first_m1/new_weighted_inf_norm1,
      //                        new_first_m2/new_weighted_inf_norm2]
      //         = [1.0, 1.0]
      // w = [ w1_old - lr*updates_1, w2_old - lr*updates_2]
      //            = [-0.2, -0.2]

      session.train(y, [{tensor: x, data: inputProvider}], 1, optimizer);
      const dydw2 = session.activationArrayMap.get(w).dataSync();
      test_util.expectArraysClose(dydw2, new Float32Array([-.2, -.2]), 2e-5);
    });
  });
});
