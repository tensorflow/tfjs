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
import {Graph} from '../../graph/graph';
import {Session} from '../../graph/session';
import {Array1D, Scalar, variable} from '../../math/ndarray';
import * as test_util from '../../test_util';
import {MathTests} from '../../test_util';

import {SGDOptimizer} from './sgd_optimizer';

const tests: MathTests = it => {
  it('basic', math => {
    const learningRate = .1;
    const optimizer = new SGDOptimizer(learningRate);

    const x = variable(Scalar.new(4));

    let numArrays = math.getNumArrays();

    let cost = optimizer.minimize(() => math.square(x), /* returnCost */ true);

    // Cost should be the only additional array.
    expect(math.getNumArrays()).toBe(numArrays + 1);

    // de/dx = 2x
    const expectedValue1 = -2 * 4 * learningRate + 4;
    test_util.expectArraysClose(x, [expectedValue1]);
    test_util.expectArraysClose(cost, [Math.pow(4, 2)]);

    cost.dispose();
    numArrays = math.getNumArrays();

    cost = optimizer.minimize(() => math.square(x), /* returnCost */ false);
    // There should be no new additional NDArrays.
    expect(math.getNumArrays()).toBe(numArrays);

    const expectedValue2 = -2 * expectedValue1 * learningRate + expectedValue1;
    test_util.expectArraysClose(x, [expectedValue2]);
    expect(cost).toBe(null);

    optimizer.dispose();
    x.dispose();
    // There should be no more NDArrays.
    expect(math.getNumArrays()).toBe(0);
  });

  it('graph', math => {
    const inputProvider: InputProvider = {
      getNextCopy() {
        return Array1D.new([2, 4]);
      },
      disposeCopy(math, example) {}
    };

    const g = new Graph();
    const x = g.placeholder('x', [2]);
    const y = g.square(x);
    const z = g.add(x, g.constant(3));
    const w = g.reduceSum(g.add(y, z));
    const optimizer = new SGDOptimizer(0.1);
    const session = new Session(g, math);
    // w = reduce_sum(x^2 + x + 3)
    // dw/dx = [2*x_1 + 1, 2*x_2 + 1]
    session.train(w, [{tensor: x, data: inputProvider}], 1, optimizer);
    const dwdx = session.gradientArrayMap.get(x).dataSync();
    test_util.expectArraysClose(dwdx, new Float32Array([5, 9]), 1e-1);
  });
};

test_util.describeMathCPU('SGDOptimizer', [tests]);
test_util.describeMathGPU('SGDOptimizer', [tests], [
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
]);
