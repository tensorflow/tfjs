/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

import * as tf from './index';
import {ALL_ENVS, describeWithFlags} from './jasmine_util';
import {Scalar, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D, Variable} from './tensor';
import {expectArraysClose} from './test_util';
import {Rank} from './types';

describeWithFlags('variable', ALL_ENVS, () => {
  it('simple assign', async () => {
    const v = tf.variable(tf.tensor1d([1, 2, 3]));
    expectArraysClose(await v.data(), [1, 2, 3]);

    v.assign(tf.tensor1d([4, 5, 6]));
    expectArraysClose(await v.data(), [4, 5, 6]);
  });

  it('simple chain assign', async () => {
    const v = tf.tensor1d([1, 2, 3]).variable();
    expectArraysClose(await v.data(), [1, 2, 3]);

    v.assign(tf.tensor1d([4, 5, 6]));
    expectArraysClose(await v.data(), [4, 5, 6]);
  });

  it('default names are unique', () => {
    const v = tf.variable(tf.tensor1d([1, 2, 3]));
    expect(v.name).not.toBeNull();

    const v2 = tf.variable(tf.tensor1d([1, 2, 3]));
    expect(v2.name).not.toBeNull();
    expect(v.name).not.toBe(v2.name);
  });

  it('user provided name', () => {
    const v = tf.variable(tf.tensor1d([1, 2, 3]), true, 'myName');
    expect(v.name).toBe('myName');
  });

  it('if name already used, throw error', () => {
    tf.variable(tf.tensor1d([1, 2, 3]), true, 'myName');
    expect(() => tf.variable(tf.tensor1d([1, 2, 3]), true, 'myName'))
        .toThrowError();
  });

  it('ops can take variables', async () => {
    const value = tf.tensor1d([1, 2, 3]);
    const v = tf.variable(value);
    const res = tf.sum(v);
    expectArraysClose(await res.data(), [6]);
  });

  it('chained variables works', async () => {
    const v = tf.tensor1d([1, 2, 3]).variable();
    const res = tf.sum(v);
    expectArraysClose(await res.data(), [6]);
  });

  it('variables are not affected by tidy', async () => {
    let v: Variable<Rank.R1>;
    expect(tf.memory().numTensors).toBe(0);

    tf.tidy(() => {
      const value = tf.tensor1d([1, 2, 3], 'float32');
      expect(tf.memory().numTensors).toBe(1);

      v = tf.variable(value);
      expect(tf.memory().numTensors).toBe(2);
    });

    expect(tf.memory().numTensors).toBe(1);
    expectArraysClose(await v.data(), [1, 2, 3]);

    v.dispose();
    expect(tf.memory().numTensors).toBe(0);
  });

  it('disposing a named variable allows creating new named variable', () => {
    const numTensors = tf.memory().numTensors;
    const t = tf.scalar(1);
    const varName = 'var';
    const v = tf.variable(t, true, varName);

    expect(tf.memory().numTensors).toBe(numTensors + 2);

    v.dispose();
    t.dispose();

    expect(tf.memory().numTensors).toBe(numTensors);

    // Create another variable with the same name.
    const t2 = tf.scalar(1);
    const v2 = tf.variable(t2, true, varName);

    expect(tf.memory().numTensors).toBe(numTensors + 2);

    t2.dispose();
    v2.dispose();

    expect(tf.memory().numTensors).toBe(numTensors);
  });

  it('double disposing a variable works', () => {
    const numTensors = tf.memory().numTensors;

    const t = tf.scalar(1);
    const v = tf.variable(t);

    expect(tf.memory().numTensors).toBe(numTensors + 2);

    t.dispose();
    v.dispose();

    expect(tf.memory().numTensors).toBe(numTensors);

    // Double dispose the variable.
    v.dispose();
    expect(tf.memory().numTensors).toBe(numTensors);
  });

  it('constructor does not dispose', async () => {
    const a = tf.scalar(2);
    const v = tf.variable(a);

    expect(tf.memory().numTensors).toBe(2);
    expect(tf.memory().numDataBuffers).toBe(1);
    expectArraysClose(await v.data(), [2]);
    expectArraysClose(await a.data(), [2]);
  });

  it('variables are assignable to tensors', () => {
    // This test asserts compilation, not doing any run-time assertion.
    const x0: Variable<Rank.R0> = null;
    const y0: Scalar = x0;
    expect(y0).toBeNull();

    const x1: Variable<Rank.R1> = null;
    const y1: Tensor1D = x1;
    expect(y1).toBeNull();

    const x2: Variable<Rank.R2> = null;
    const y2: Tensor2D = x2;
    expect(y2).toBeNull();

    const x3: Variable<Rank.R3> = null;
    const y3: Tensor3D = x3;
    expect(y3).toBeNull();

    const x4: Variable<Rank.R4> = null;
    const y4: Tensor4D = x4;
    expect(y4).toBeNull();

    const xh: Variable = null;
    const yh: Tensor = xh;
    expect(yh).toBeNull();
  });

  it('assign does not dispose old data', async () => {
    let v: Variable<Rank.R1>;
    v = tf.variable(tf.tensor1d([1, 2, 3]));

    expect(tf.memory().numTensors).toBe(2);
    expect(tf.memory().numDataBuffers).toBe(1);

    expectArraysClose(await v.data(), [1, 2, 3]);

    const secondArray = tf.tensor1d([4, 5, 6]);
    expect(tf.memory().numTensors).toBe(3);
    expect(tf.memory().numDataBuffers).toBe(2);

    v.assign(secondArray);
    expectArraysClose(await v.data(), [4, 5, 6]);
    // Assign doesn't dispose the 1st array.
    expect(tf.memory().numTensors).toBe(3);
    expect(tf.memory().numDataBuffers).toBe(2);

    v.dispose();
    // Disposing the variable disposes itself. The input to variable and
    // secondArray are the only remaining tensors.
    expect(tf.memory().numTensors).toBe(2);
    expect(tf.memory().numDataBuffers).toBe(2);
  });

  it('shape must match', () => {
    const v = tf.variable(tf.tensor1d([1, 2, 3]));
    expect(() => v.assign(tf.tensor1d([1, 2]))).toThrowError();
    // tslint:disable-next-line:no-any
    expect(() => v.assign(tf.tensor2d([3, 4], [1, 2]) as any)).toThrowError();
  });

  it('dtype must match', () => {
    const v = tf.variable(tf.tensor1d([1, 2, 3]));
    // tslint:disable-next-line:no-any
    expect(() => v.assign(tf.tensor1d([1, 1, 1], 'int32') as any))
        .toThrowError();
    // tslint:disable-next-line:no-any
    expect(() => v.assign(tf.tensor1d([true, false, true], 'bool') as any))
        .toThrowError();
  });
});

describeWithFlags('x instanceof Variable', ALL_ENVS, () => {
  it('x: Variable', () => {
    const t = tf.variable(tf.scalar(1));
    expect(t instanceof Variable).toBe(true);
  });

  it('x: other object, fails', () => {
    const t = {something: 'else'};
    expect(t instanceof Variable).toBe(false);
  });

  it('x: Tensor, fails', () => {
    const t = tf.scalar(1);
    expect(t instanceof Variable).toBe(false);
  });
});
