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

import * as dl from '../../index';
import {variable, Variable} from '../../math/tensor';
import * as test_util from '../../test_util';
import {MathTests} from '../../test_util';
import {SGDOptimizer} from './sgd_optimizer';

const tests: MathTests = it => {
  it('basic', math => {
    const learningRate = .1;
    const optimizer = new SGDOptimizer(learningRate);

    const x = variable(dl.scalar(4));
    const bias = variable(dl.scalar(1));
    const strayVariable = variable(dl.scalar(-1));

    let numTensors = math.getNumTensors();

    const f = () => math.addStrict(math.square(x), bias);

    let cost = optimizer.minimize(f, /* returnCost */ true);

    // Cost should be the only additional array.
    expect(math.getNumTensors()).toBe(numTensors + 1);

    // de/dx = 2x
    const expectedX1 = -2 * 4 * learningRate + 4;
    // de/db = 1
    const expectedBias1 = -1 * learningRate + 1;
    test_util.expectArraysClose(x, [expectedX1]);
    test_util.expectArraysClose(bias, [expectedBias1]);
    test_util.expectArraysClose(cost, [Math.pow(4, 2) + 1]);
    // The stray variable should remain unchanged.
    test_util.expectArraysClose(strayVariable, [-1]);

    cost.dispose();
    numTensors = math.getNumTensors();

    cost = optimizer.minimize(f, /* returnCost */ false);
    // There should be no new additional Tensors.
    expect(math.getNumTensors()).toBe(numTensors);

    const expectedX2 = -2 * expectedX1 * learningRate + expectedX1;
    const expectedBias2 = -learningRate + expectedBias1;
    test_util.expectArraysClose(x, [expectedX2]);
    test_util.expectArraysClose(bias, [expectedBias2]);
    expect(cost).toBe(null);
    // The stray variable should remain unchanged.
    test_util.expectArraysClose(strayVariable, [-1]);

    optimizer.dispose();
    x.dispose();
    bias.dispose();
    strayVariable.dispose();
    // There should be no more Tensors.
    expect(math.getNumTensors()).toBe(0);
  });

  it('varList array of all variables', math => {
    const learningRate = .1;
    const optimizer = new SGDOptimizer(learningRate);

    const x = variable(dl.scalar(4));
    const bias = variable(dl.scalar(1));
    const strayVariable = variable(dl.scalar(-1));
    const varList = [x, bias];

    const f = () => math.addStrict(math.square(x), bias);

    let cost = optimizer.minimize(f, /* returnCost */ true, varList);

    // de/dx = 2x
    const expectedX1 = -2 * 4 * learningRate + 4;
    // de/db = 1
    const expectedBias1 = -1 * learningRate + 1;
    test_util.expectArraysClose(x, [expectedX1]);
    test_util.expectArraysClose(bias, [expectedBias1]);
    test_util.expectArraysClose(cost, [Math.pow(4, 2) + 1]);
    // The stray variable should remain unchanged.
    test_util.expectArraysClose(strayVariable, [-1]);

    cost = optimizer.minimize(f, /* returnCost */ false, varList);

    const expectedX2 = -2 * expectedX1 * learningRate + expectedX1;
    const expectedBias2 = -learningRate + expectedBias1;
    test_util.expectArraysClose(x, [expectedX2]);
    test_util.expectArraysClose(bias, [expectedBias2]);
    // The stray variable should remain unchanged.
    test_util.expectArraysClose(strayVariable, [-1]);
    expect(cost).toBe(null);
  });

  it('varList empty array of variables to update updates nothing', math => {
    const learningRate = .1;
    const optimizer = new SGDOptimizer(learningRate);

    const x = variable(dl.scalar(4));
    const bias = variable(dl.scalar(1));
    const strayVariable = variable(dl.scalar(-1));
    const varList: Variable[] = [];

    const f = () => math.addStrict(math.square(x), bias);

    let cost = optimizer.minimize(f, /* returnCost */ true, varList);

    // x should not have been updated.
    test_util.expectArraysClose(x, [4]);
    // bias should not have been updated.
    test_util.expectArraysClose(bias, [1]);
    test_util.expectArraysClose(cost, [Math.pow(4, 2) + 1]);
    // The stray variable should remain unchanged.
    test_util.expectArraysClose(strayVariable, [-1]);

    cost = optimizer.minimize(f, /* returnCost */ false, varList);

    // x again should not have been updated.
    test_util.expectArraysClose(x, [4]);
    // bias again should not have been updated.
    test_util.expectArraysClose(bias, [1]);
    // The stray variable should remain unchanged.
    test_util.expectArraysClose(strayVariable, [-1]);
    expect(cost).toBe(null);
  });

  it('varList subset of variables update', math => {
    const learningRate = .1;
    const optimizer = new SGDOptimizer(learningRate);

    const x = variable(dl.scalar(4));
    const bias = variable(dl.scalar(1));
    const strayVariable = variable(dl.scalar(-1));
    const varList = [x];

    const f = () => math.addStrict(math.square(x), bias);

    let cost = optimizer.minimize(f, /* returnCost */ true, varList);

    // de/dx = 2x
    const expectedValue1 = -2 * 4 * learningRate + 4;
    test_util.expectArraysClose(x, [expectedValue1]);
    // bias should remain unchanged.
    test_util.expectArraysClose(bias, [1]);
    test_util.expectArraysClose(cost, [Math.pow(4, 2) + 1]);
    // The stray variable should remain unchanged.
    test_util.expectArraysClose(strayVariable, [-1]);

    cost = optimizer.minimize(f, /* returnCost */ false, varList);

    const expectedValue2 = -2 * expectedValue1 * learningRate + expectedValue1;
    test_util.expectArraysClose(x, [expectedValue2]);
    // Bias still should remain unchanged.
    test_util.expectArraysClose(bias, [1]);
    expect(cost).toBe(null);
    // The stray variable should remain unchanged.
    test_util.expectArraysClose(strayVariable, [-1]);
  });

  it('only bias trainable', math => {
    const learningRate = .1;
    const optimizer = new SGDOptimizer(learningRate);

    const trainable = false;
    const x = variable(dl.scalar(4), trainable);
    const bias = variable(dl.scalar(1));
    const strayVariable = variable(dl.scalar(-1));

    const f = () => math.addStrict(math.square(x), bias);

    let cost = optimizer.minimize(f, /* returnCost */ true);

    // x should not have been updated.
    test_util.expectArraysClose(x, [4]);
    // de/db = 1
    const expectedBias1 = -1 * learningRate + 1;
    test_util.expectArraysClose(bias, [expectedBias1]);
    test_util.expectArraysClose(cost, [Math.pow(4, 2) + 1]);
    // The stray variable should remain unchanged.
    test_util.expectArraysClose(strayVariable, [-1]);

    cost = optimizer.minimize(f, /* returnCost */ false);

    // x should not have been updated.
    test_util.expectArraysClose(x, [4]);
    const expectedBias2 = -learningRate + expectedBias1;
    test_util.expectArraysClose(bias, [expectedBias2]);
    expect(cost).toBe(null);
    // The stray variable should remain unchanged.
    test_util.expectArraysClose(strayVariable, [-1]);
  });

  it('only bias trainable, only x in varList does nothing', math => {
    const learningRate = .1;
    const optimizer = new SGDOptimizer(learningRate);

    const trainable = false;
    const x = variable(dl.scalar(4), trainable);
    const bias = variable(dl.scalar(1));
    const strayVariable = variable(dl.scalar(-1));
    const varList = [x];

    const f = () => math.addStrict(math.square(x), bias);

    let cost = optimizer.minimize(f, /* returnCost */ true, varList);

    // x should not have been updated.
    test_util.expectArraysClose(x, [4]);
    // bias should remain unchanged.
    test_util.expectArraysClose(bias, [1]);
    test_util.expectArraysClose(cost, [Math.pow(4, 2) + 1]);
    // The stray variable should remain unchanged.
    test_util.expectArraysClose(strayVariable, [-1]);

    cost = optimizer.minimize(f, /* returnCost */ false, varList);

    // x should not have been updated.
    test_util.expectArraysClose(x, [4]);
    // bias should not have been updated.
    test_util.expectArraysClose(bias, [1]);
    expect(cost).toBe(null);
    // The stray variable should remain unchanged.
    test_util.expectArraysClose(strayVariable, [-1]);
  });

  it('throws error when f returns a non-scalar', math => {
    const learningRate = .1;
    const optimizer = new SGDOptimizer(learningRate);

    const x = variable(dl.tensor1d([1, 2]));
    const f = () => x.square();

    // tslint:disable-next-line:no-any
    expect(() => optimizer.minimize(f as any)).toThrowError();
  });
};

test_util.describeMathCPU('Optimizer', [tests]);
test_util.describeMathGPU('Optimizer', [tests], [
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
  {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
]);
