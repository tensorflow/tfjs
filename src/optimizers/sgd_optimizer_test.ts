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

import * as dl from '../index';
import {ALL_ENVS, describeWithFlags, expectArraysClose} from '../test_util';

describeWithFlags('SGDOptimizer', ALL_ENVS, () => {
  it('basic', () => {
    const learningRate = .1;
    const optimizer = dl.train.sgd(learningRate);

    const x = dl.scalar(4).variable();

    let numTensors = dl.memory().numTensors;

    let cost = optimizer.minimize(() => x.square(), /* returnCost */ true);

    // Cost should be the only additional array.
    expect(dl.memory().numTensors).toBe(numTensors + 1);

    // de/dx = 2x
    const expectedValue1 = -2 * 4 * learningRate + 4;
    expectArraysClose(x, [expectedValue1]);
    expectArraysClose(cost, [Math.pow(4, 2)]);

    cost.dispose();
    numTensors = dl.memory().numTensors;

    cost = optimizer.minimize(() => x.square(), /* returnCost */ false);
    // There should be no new additional Tensors.
    expect(dl.memory().numTensors).toBe(numTensors);

    const expectedValue2 = -2 * expectedValue1 * learningRate + expectedValue1;
    expectArraysClose(x, [expectedValue2]);
    expect(cost).toBe(null);

    optimizer.dispose();
    x.dispose();
    // The only tensor remaining is the argument to variable().
    expect(dl.memory().numTensors).toBe(1);
  });
});
