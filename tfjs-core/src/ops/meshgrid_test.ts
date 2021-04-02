/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF {} KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '../index';
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {Tensor} from '../tensor';
import {expectArraysEqual, expectArraysClose} from '../test_util';

describeWithFlags('no input', ALL_ENVS, () => {
  it('Should return an empty tensor ', async () => {
    expect(tf.meshgrid()).toEqual([]);
  });
});

describeWithFlags('single input', ALL_ENVS, () => {
  it('Should return a tensor with the same data', async () => {
    const x = [1, 2, 3, 4];
    const [got] = tf.meshgrid(x);

    expectArraysEqual(await got.data(), x);
  });
});

describeWithFlags('simple inputs', ALL_ENVS, () => {
  it('Should handle the simple 2D case', async () => {
    const x = [1, 2, 3];
    const y = [4, 5, 6, 7];
    const [X, Y] = tf.meshgrid(x, y);

    // 'close' instead of 'equal' because of matmul precision
    // in certain backends (WebGL).
    expectArraysClose(
        await X.data(), [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]);
    expectArraysClose(
        await Y.data(), [[4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]]);
  });

  it('Should support \'ij\' indexing', async () => {
    const x = [1, 2, 3];
    const y = [4, 5, 6, 7];
    const [X, Y] = tf.meshgrid(x, y, {indexing: 'ij'});

    // 'close' instead of 'equal' because of matmul precision
    // in certain backends (WebGL).
    expectArraysClose(
        await X.data(), [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]);
    expectArraysClose(
        await Y.data(), [[4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7]]);
  });
});

describeWithFlags('higher dimensional input', ALL_ENVS, () => {
  it('Should flatten higher dimensional', async () => {
    const x = [1, 2, 3];
    const a = [[1, 1], [1, 1]];

    const [X, A] = tf.meshgrid(x, a);

    // 'close' instead of 'equal' because of matmul precision
    // in certain backends (WebGL).
    expectArraysClose(
        await X.data(), [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]);
    expectArraysClose(
        await A.data(), [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]);
  });
});

describeWithFlags('dtypes', ALL_ENVS, () => {
  it('Should use float32 for arrays of numbers', async () => {
    const x = [1, 2];
    const y = [3, 4];
    const [X, Y] = tf.meshgrid(x, y);

    expect(X.dtype).toBe('float32');
    expect(Y.dtype).toBe('float32');
  });

  it('Should use the input tensor dtype', async () => {
    const x = tf.tensor1d([1, 2], 'int32');
    const y = tf.tensor1d([3, 4], 'float32');
    const [X, Y] = tf.meshgrid(x, y);

    expect(X.dtype).toBe('int32');
    expect(Y.dtype).toBe('float32');
  });
});

describeWithFlags('scalars', ALL_ENVS, () => {
  it('Should treat them as 1D tensors', async () => {
    const [X] = tf.meshgrid(0);
    // 'close' instead of 'equal' because of matmul precision
    // in certain backends (WebGL).
    expectArraysClose(await X.data(), [0]);

    const [Y, Z] = tf.meshgrid([0], 1);
    expectArraysClose(await Y.data(), [[0]]);
    expectArraysClose(await Z.data(), [[1]]);
  });
});

describeWithFlags('invalid arguments', ALL_ENVS, () => {
  it('Should throw an Error', () => {
    expect(() => tf.meshgrid((() => {}) as {} as Tensor)).toThrow();
    expect(() => tf.meshgrid([1], (() => {}) as {} as Tensor)).toThrow();
    expect(() => tf.meshgrid([1], [2], {indexing: 'foobar'})).toThrow();
  });
});
