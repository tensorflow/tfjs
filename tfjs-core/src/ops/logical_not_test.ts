/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import {expectArraysClose} from '../test_util';

describeWithFlags('logicalNot', ALL_ENVS, () => {
  it('Tensor1D.', async () => {
    let a = tf.tensor1d([1, 0, 0], 'bool');
    expectArraysClose(await tf.logicalNot(a).data(), [0, 1, 1]);

    a = tf.tensor1d([0, 0, 0], 'bool');
    expectArraysClose(await tf.logicalNot(a).data(), [1, 1, 1]);

    a = tf.tensor1d([1, 1], 'bool');
    expectArraysClose(await tf.logicalNot(a).data(), [0, 0]);
  });
  it('Tests chaining in Tensor1D', async () => {
    let a = tf.tensor1d([1, 0, 0], 'bool');
    expectArraysClose(await a.logicalNot().data(), [0, 1, 1]);

    a = tf.tensor1d([0, 0, 0], 'bool');
    expectArraysClose(await a.logicalNot().data(), [1, 1, 1]);

    a = tf.tensor1d([1, 1], 'bool');
    expectArraysClose(await a.logicalNot().data(), [0, 0]);
  });

  it('Tensor2D', async () => {
    let a = tf.tensor2d([[1, 0, 1], [0, 0, 0]], [2, 3], 'bool');
    expectArraysClose(await tf.logicalNot(a).data(), [0, 1, 0, 1, 1, 1]);

    a = tf.tensor2d([[0, 0, 0], [1, 1, 1]], [2, 3], 'bool');
    expectArraysClose(await tf.logicalNot(a).data(), [1, 1, 1, 0, 0, 0]);
  });

  it('Tensor3D', async () => {
    let a = tf.tensor3d([[[1], [0], [1]], [[0], [0], [0]]], [2, 3, 1], 'bool');
    expectArraysClose(await tf.logicalNot(a).data(), [0, 1, 0, 1, 1, 1]);

    a = tf.tensor3d([[[0], [0], [0]], [[1], [1], [1]]], [2, 3, 1], 'bool');
    expectArraysClose(await tf.logicalNot(a).data(), [1, 1, 1, 0, 0, 0]);
  });

  it('Tensor4D', async () => {
    let a = tf.tensor4d([1, 0, 1, 0], [2, 2, 1, 1], 'bool');
    expectArraysClose(await tf.logicalNot(a).data(), [0, 1, 0, 1]);

    a = tf.tensor4d([0, 0, 0, 0], [2, 2, 1, 1], 'bool');
    expectArraysClose(await tf.logicalNot(a).data(), [1, 1, 1, 1]);

    a = tf.tensor4d([1, 1, 1, 1], [2, 2, 1, 1], 'bool');
    expectArraysClose(await tf.logicalNot(a).data(), [0, 0, 0, 0]);
  });

  it('Tensor6D', async () => {
    let a = tf.tensor6d([1, 0, 1, 0], [2, 2, 1, 1, 1, 1], 'bool');
    expectArraysClose(await tf.logicalNot(a).data(), [0, 1, 0, 1]);

    a = tf.zeros([2, 2, 2, 2, 2, 2]).cast('bool');
    let expectedResult = new Uint8Array(64).fill(1);
    expectedResult = expectedResult.fill(1);
    expectArraysClose(await tf.logicalNot(a).data(), expectedResult);

    a = tf.ones([2, 2, 2, 2, 2, 2]).cast('bool');
    expectedResult = expectedResult.fill(0);
    expectArraysClose(await tf.logicalNot(a).data(), expectedResult);
  });

  it('throws when passed a non-tensor', () => {
    expect(() => tf.logicalNot({} as tf.Tensor))
        .toThrowError(/Argument 'x' passed to 'logicalNot' must be a Tensor/);
  });

  it('accepts a tensor-like object', async () => {
    const a = [1, 0, 0];
    expectArraysClose(await tf.logicalNot(a).data(), [0, 1, 1]);
  });
});
