/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import * as tf from '../index';
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {Variable} from '../tensor';
import {expectArraysClose} from '../test_util';

import {Optimizer} from './optimizer';
import {SGDOptimizer} from './sgd_optimizer';

describeWithFlags('optimizer', ALL_ENVS, () => {
  it('basic', async () => {
    const initialTensors = tf.memory().numTensors;
    const learningRate = .1;
    const optimizer = tf.train.sgd(learningRate);

    const x = tf.scalar(4).variable();
    const bias = tf.scalar(1).variable();
    const strayVariable = tf.scalar(-1).variable();

    let numTensors = tf.memory().numTensors;

    // tslint:disable-next-line: no-unnecessary-type-assertion
    const f = () => x.square().add(bias) as tf.Scalar;

    let cost = optimizer.minimize(f, /* returnCost */ true);

    // Cost should be the only additional arrays.
    expect(tf.memory().numTensors).toBe(numTensors + 1);

    // de/dx = 2x
    const expectedX1 = -2 * 4 * learningRate + 4;
    // de/db = 1
    const expectedBias1 = -1 * learningRate + 1;
    expectArraysClose(await x.data(), [expectedX1]);
    expectArraysClose(await bias.data(), [expectedBias1]);
    expectArraysClose(await cost.data(), [Math.pow(4, 2) + 1]);
    // The stray variable should remain unchanged.
    expectArraysClose(await strayVariable.data(), [-1]);

    cost.dispose();
    numTensors = tf.memory().numTensors;

    cost = optimizer.minimize(f, /* returnCost */ false);
    // There should be no new additional Tensors.
    expect(tf.memory().numTensors).toBe(numTensors);

    const expectedX2 = -2 * expectedX1 * learningRate + expectedX1;
    const expectedBias2 = -learningRate + expectedBias1;
    expectArraysClose(await x.data(), [expectedX2]);
    expectArraysClose(await bias.data(), [expectedBias2]);
    expect(cost).toBe(null);
    // The stray variable should remain unchanged.
    expectArraysClose(await strayVariable.data(), [-1]);

    optimizer.dispose();
    x.dispose();
    bias.dispose();
    strayVariable.dispose();
    // The only additional tensors remaining are the arguments to variable().
    expect(tf.memory().numTensors).toBe(initialTensors + 3);
  });

  it('varList array of all variables', async () => {
    const learningRate = .1;
    const optimizer = new SGDOptimizer(learningRate);

    const x = tf.scalar(4).variable();
    const bias = tf.scalar(1).variable();
    const strayVariable = tf.scalar(-1).variable();
    const varList = [x, bias];

    // tslint:disable-next-line: no-unnecessary-type-assertion
    const f = () => x.square().add(bias) as tf.Scalar;

    let cost = optimizer.minimize(f, /* returnCost */ true, varList);

    // de/dx = 2x
    const expectedX1 = -2 * 4 * learningRate + 4;
    // de/db = 1
    const expectedBias1 = -1 * learningRate + 1;
    expectArraysClose(await x.data(), [expectedX1]);
    expectArraysClose(await bias.data(), [expectedBias1]);
    expectArraysClose(await cost.data(), [Math.pow(4, 2) + 1]);
    // The stray variable should remain unchanged.
    expectArraysClose(await strayVariable.data(), [-1]);

    cost = optimizer.minimize(f, /* returnCost */ false, varList);

    const expectedX2 = -2 * expectedX1 * learningRate + expectedX1;
    const expectedBias2 = -learningRate + expectedBias1;
    expectArraysClose(await x.data(), [expectedX2]);
    expectArraysClose(await bias.data(), [expectedBias2]);
    // The stray variable should remain unchanged.
    expectArraysClose(await strayVariable.data(), [-1]);
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

    // tslint:disable-next-line: no-unnecessary-type-assertion
    const f = () => x.square().add(bias) as tf.Scalar;

    expect(() => optimizer.minimize(f, /* returnCost */ true, varList))
        .toThrowError();
  });

  it('varList subset of variables update', async () => {
    const learningRate = .1;
    const optimizer = new SGDOptimizer(learningRate);

    const x = tf.scalar(4).variable();
    const bias = tf.scalar(1).variable();
    const strayVariable = tf.scalar(-1).variable();
    const varList = [x];

    // tslint:disable-next-line: no-unnecessary-type-assertion
    const f = () => x.square().add(bias) as tf.Scalar;

    let cost = optimizer.minimize(f, /* returnCost */ true, varList);

    // de/dx = 2x
    const expectedValue1 = -2 * 4 * learningRate + 4;
    expectArraysClose(await x.data(), [expectedValue1]);
    // bias should remain unchanged.
    expectArraysClose(await bias.data(), [1]);
    expectArraysClose(await cost.data(), [Math.pow(4, 2) + 1]);
    // The stray variable should remain unchanged.
    expectArraysClose(await strayVariable.data(), [-1]);

    cost = optimizer.minimize(f, /* returnCost */ false, varList);

    const expectedValue2 = -2 * expectedValue1 * learningRate + expectedValue1;
    expectArraysClose(await x.data(), [expectedValue2]);
    // Bias still should remain unchanged.
    expectArraysClose(await bias.data(), [1]);
    expect(cost).toBe(null);
    // The stray variable should remain unchanged.
    expectArraysClose(await strayVariable.data(), [-1]);
  });

  it('only bias trainable', async () => {
    const learningRate = .1;
    const optimizer = new SGDOptimizer(learningRate);

    const trainable = false;
    const x = tf.scalar(4).variable(trainable);
    const bias = tf.scalar(1).variable();
    const strayVariable = tf.scalar(-1).variable();

    // tslint:disable-next-line: no-unnecessary-type-assertion
    const f = () => x.square().add(bias) as tf.Scalar;

    let cost = optimizer.minimize(f, /* returnCost */ true);

    // x should not have been updated.
    expectArraysClose(await x.data(), [4]);
    // de/db = 1
    const expectedBias1 = -1 * learningRate + 1;
    expectArraysClose(await bias.data(), [expectedBias1]);
    expectArraysClose(await cost.data(), [Math.pow(4, 2) + 1]);
    // The stray variable should remain unchanged.
    expectArraysClose(await strayVariable.data(), [-1]);

    cost = optimizer.minimize(f, /* returnCost */ false);

    // x should not have been updated.
    expectArraysClose(await x.data(), [4]);
    const expectedBias2 = -learningRate + expectedBias1;
    expectArraysClose(await bias.data(), [expectedBias2]);
    expect(cost).toBe(null);
    // The stray variable should remain unchanged.
    expectArraysClose(await strayVariable.data(), [-1]);
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

    // tslint:disable-next-line: no-unnecessary-type-assertion
    const f = () => x.square().add(bias) as tf.Scalar;

    expect(() => optimizer.minimize(f, /* returnCost */ true, varList))
        .toThrowError();
  });

  it('instanceof Optimizer', () => {
    const learningRate = .1;
    const optimizer = new SGDOptimizer(learningRate);

    expect(optimizer instanceof Optimizer).toBe(true);
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
