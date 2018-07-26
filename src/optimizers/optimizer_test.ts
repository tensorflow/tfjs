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

import * as tf from '..';
import {describeWithFlags} from '../jasmine_util';
import {Variable} from '../tensor';
import {ALL_ENVS, expectArraysClose} from '../test_util';
import {SGDOptimizer} from './sgd_optimizer';

describeWithFlags('optimizer', ALL_ENVS, () => {
  it('basic', () => {
    const learningRate = .1;
    const optimizer = tf.train.sgd(learningRate);

    const x = tf.scalar(4).variable();
    const bias = tf.scalar(1).variable();
    const strayVariable = tf.scalar(-1).variable();

    let numTensors = tf.memory().numTensors;

    const f = () => x.square().addStrict(bias);

    let cost = optimizer.minimize(f, /* returnCost */ true);

    // Cost should be the only additional array.
    expect(tf.memory().numTensors).toBe(numTensors + 1);

    // de/dx = 2x
    const expectedX1 = -2 * 4 * learningRate + 4;
    // de/db = 1
    const expectedBias1 = -1 * learningRate + 1;
    expectArraysClose(x, [expectedX1]);
    expectArraysClose(bias, [expectedBias1]);
    expectArraysClose(cost, [Math.pow(4, 2) + 1]);
    // The stray variable should remain unchanged.
    expectArraysClose(strayVariable, [-1]);

    cost.dispose();
    numTensors = tf.memory().numTensors;

    cost = optimizer.minimize(f, /* returnCost */ false);
    // There should be no new additional Tensors.
    expect(tf.memory().numTensors).toBe(numTensors);

    const expectedX2 = -2 * expectedX1 * learningRate + expectedX1;
    const expectedBias2 = -learningRate + expectedBias1;
    expectArraysClose(x, [expectedX2]);
    expectArraysClose(bias, [expectedBias2]);
    expect(cost).toBe(null);
    // The stray variable should remain unchanged.
    expectArraysClose(strayVariable, [-1]);

    optimizer.dispose();
    x.dispose();
    bias.dispose();
    strayVariable.dispose();
    // The only tensors remaining are the arguments to variable().
    expect(tf.memory().numTensors).toBe(3);
  });

  it('varList array of all variables', () => {
    const learningRate = .1;
    const optimizer = new SGDOptimizer(learningRate);

    const x = tf.scalar(4).variable();
    const bias = tf.scalar(1).variable();
    const strayVariable = tf.scalar(-1).variable();
    const varList = [x, bias];

    const f = () => x.square().addStrict(bias);

    let cost = optimizer.minimize(f, /* returnCost */ true, varList);

    // de/dx = 2x
    const expectedX1 = -2 * 4 * learningRate + 4;
    // de/db = 1
    const expectedBias1 = -1 * learningRate + 1;
    expectArraysClose(x, [expectedX1]);
    expectArraysClose(bias, [expectedBias1]);
    expectArraysClose(cost, [Math.pow(4, 2) + 1]);
    // The stray variable should remain unchanged.
    expectArraysClose(strayVariable, [-1]);

    cost = optimizer.minimize(f, /* returnCost */ false, varList);

    const expectedX2 = -2 * expectedX1 * learningRate + expectedX1;
    const expectedBias2 = -learningRate + expectedBias1;
    expectArraysClose(x, [expectedX2]);
    expectArraysClose(bias, [expectedBias2]);
    // The stray variable should remain unchanged.
    expectArraysClose(strayVariable, [-1]);
    expect(cost).toBe(null);
  });

  it('varList empty array of variables throws error', () => {
    const learningRate = .1;
    const optimizer = new SGDOptimizer(learningRate);

    const x = tf.scalar(4).variable();
    const bias = tf.scalar(1).variable();
    // Stray variable.
    tf.scalar(-1).variable();
    const varList: Variable[] = [];

    const f = () => x.square().addStrict(bias);

    expect(() => optimizer.minimize(f, /* returnCost */ true, varList))
        .toThrowError();
  });

  it('varList subset of variables update', () => {
    const learningRate = .1;
    const optimizer = new SGDOptimizer(learningRate);

    const x = tf.scalar(4).variable();
    const bias = tf.scalar(1).variable();
    const strayVariable = tf.scalar(-1).variable();
    const varList = [x];

    const f = () => x.square().addStrict(bias);

    let cost = optimizer.minimize(f, /* returnCost */ true, varList);

    // de/dx = 2x
    const expectedValue1 = -2 * 4 * learningRate + 4;
    expectArraysClose(x, [expectedValue1]);
    // bias should remain unchanged.
    expectArraysClose(bias, [1]);
    expectArraysClose(cost, [Math.pow(4, 2) + 1]);
    // The stray variable should remain unchanged.
    expectArraysClose(strayVariable, [-1]);

    cost = optimizer.minimize(f, /* returnCost */ false, varList);

    const expectedValue2 = -2 * expectedValue1 * learningRate + expectedValue1;
    expectArraysClose(x, [expectedValue2]);
    // Bias still should remain unchanged.
    expectArraysClose(bias, [1]);
    expect(cost).toBe(null);
    // The stray variable should remain unchanged.
    expectArraysClose(strayVariable, [-1]);
  });

  it('only bias trainable', () => {
    const learningRate = .1;
    const optimizer = new SGDOptimizer(learningRate);

    const trainable = false;
    const x = tf.scalar(4).variable(trainable);
    const bias = tf.scalar(1).variable();
    const strayVariable = tf.scalar(-1).variable();

    const f = () => x.square().addStrict(bias);

    let cost = optimizer.minimize(f, /* returnCost */ true);

    // x should not have been updated.
    expectArraysClose(x, [4]);
    // de/db = 1
    const expectedBias1 = -1 * learningRate + 1;
    expectArraysClose(bias, [expectedBias1]);
    expectArraysClose(cost, [Math.pow(4, 2) + 1]);
    // The stray variable should remain unchanged.
    expectArraysClose(strayVariable, [-1]);

    cost = optimizer.minimize(f, /* returnCost */ false);

    // x should not have been updated.
    expectArraysClose(x, [4]);
    const expectedBias2 = -learningRate + expectedBias1;
    expectArraysClose(bias, [expectedBias2]);
    expect(cost).toBe(null);
    // The stray variable should remain unchanged.
    expectArraysClose(strayVariable, [-1]);
  });

  it('only bias trainable, only x in varList throws error', () => {
    const learningRate = .1;
    const optimizer = new SGDOptimizer(learningRate);

    const trainable = false;
    const x = tf.scalar(4).variable(trainable);
    const bias = tf.scalar(1).variable();
    // stray variable.
    tf.scalar(-1).variable();
    const varList = [x];

    const f = () => x.square().addStrict(bias);

    expect(() => optimizer.minimize(f, /* returnCost */ true, varList))
        .toThrowError();
  });

  it('throws error when f returns a non-scalar', () => {
    const learningRate = .1;
    const optimizer = new SGDOptimizer(learningRate);

    const x = tf.tensor1d([1, 2]).variable();
    const f = () => x.square();

    // tslint:disable-next-line:no-any
    expect(() => optimizer.minimize(f as any)).toThrowError();
  });
});
