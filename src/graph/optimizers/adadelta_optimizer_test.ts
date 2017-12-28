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
import {AdadeltaOptimizer} from './adadelta_optimizer';

describe('adadelta optimizer', () => {
  it('basic', () => {
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
      const session = new Session(g, math);
      const optimizer = new AdadeltaOptimizer(0.1, 0.8);
      // w = reduce_sum(w_1*x_1 + w_2*x_2 + b)
      // cache = [gamma*old_cache_w1 + (1-gamma)*grad_w1**2,
      //            gamma*old_cache_w2 + (1-gamma)*grad_w2**2]
      //            = [.8, 3.2]
      // updates = [sqrt(old_updates_w1 + eps)/sqrt(old_cache_w1 + eps)*grad_w1,
      //            sqrt(old_updates_w2 + eps)/sqrT(old_cache_w2 + eps)*grad_w2]
      //            = [2, 4]
      // w = [ w1_old - lr*updates_w1,
      //            w2_old - lr*updates_w2]
      //            = [-0.2, -0.4]
      // new_updates = [gamma * old_updates_w1 + (1 - gamma) * 2**2,
      //                gamma * old_updates_w2 + (1 - gamma) * 4**2]
      //             = [0.8, 3.2]
      //
      session.train(y, [{tensor: x, data: inputProvider}], 1, optimizer);
      const dydw = session.activationArrayMap.get(w).dataSync();
      test_util.expectArraysClose(dydw, new Float32Array([-0.2, -0.4]), 1e-5);
      // cache = [gamma*old_cache_w1 + (1-gamma)*grad_w1**2,
      //            gamma*old_cache_w2 + (1-gamma)*grad_w2**2]
      //            = [1.44, 5.76]
      // updates = [sqrt(old_updates_w1 + eps)/sqrt(old_cache_w1 + eps)*grad_w1,
      //            sqrt(old_updates_w2 + eps)/sqrT(old_cache_w2 + eps)*grad_w2]
      //            = [2, 4]
      // w = [ w1_old - lr*updates_w1,
      //            w2_old - lr*updates_w2]
      //            = [-0.4, -0.8]
      session.train(y, [{tensor: x, data: inputProvider}], 1, optimizer);
      const dydw2 = session.activationArrayMap.get(w).dataSync();
      test_util.expectArraysClose(dydw2, new Float32Array([-.4, -.8]), 2e-5);
    });
  });
});
