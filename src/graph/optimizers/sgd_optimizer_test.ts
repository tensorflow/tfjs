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
import {NDArrayMathCPU} from '../../math/backends/backend_cpu';
import {Array1D} from '../../math/ndarray';
import * as test_util from '../../test_util';
import {Graph} from '../graph';
import {Session} from '../session';
import {SGDOptimizer} from './sgd_optimizer';

describe('rmsprop optimizer', () => {
  it('basic', () => {
    const g = new Graph();

    const x = g.placeholder('x', [2]);
    const y = g.square(x);
    const z = g.add(x, g.constant(3));
    const w = g.reduceSum(g.add(y, z));

    const safeMode = true;
    const optimizer = new SGDOptimizer(0.1);
    const math = new NDArrayMathCPU(safeMode);
    const session = new Session(g, math);
    const inputProvider: InputProvider = {
      getNextCopy() {
        return Array1D.new([2, 4]);
      },
      disposeCopy(math, example) {}
    };

    math.scope(() => {
      // w = reduce_sum(x^2 + x + 3)
      // dw/dx = [2*x_1 + 1, 2*x_2 + 1]
      session.train(w, [{tensor: x, data: inputProvider}], 1, optimizer);
      const dwdx = session.gradientArrayMap.get(x).getValues();
      test_util.expectArraysClose(dwdx, new Float32Array([5, 9]));
    });
  });
});
