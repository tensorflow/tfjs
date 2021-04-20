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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '../../index';
import {ALL_ENVS, describeWithFlags} from '../../jasmine_util';
import {expectArraysClose} from '../../test_util';

function sparseTensorValue5x6() {
  const ind = tf.tensor2d(
      [[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]], [6, 2], 'int32');
  const val = [0, 10, 13, 14, 32, 33];
  const shape = [5, 6];
  return {ind, val, shape};
}

function sparseTensorValue2x3x4() {
  const ind = tf.tensor2d(
      [
        [0, 0, 1], [0, 1, 0], [0, 1, 2], [1, 0, 3], [1, 1, 1], [1, 1, 3],
        [1, 2, 2]
      ],
      [7, 3], 'int32');
  const val = [1, 10, 12, 103, 111, 113, 122];
  const shape = [2, 3, 4];
  return {ind, val, shape};
}
describeWithFlags('sparseReshape', ALL_ENVS, () => {
  it('preserve static shape info', async () => {
    const sparseTensor = sparseTensorValue5x6();
    const result = tf.sparse.sparseReshape(
        sparseTensor.ind, sparseTensor.shape, [1, 5, 2, 3]);
    expectArraysClose(await result.outputShape.data(), [1, 5, 2, 3]);
  });

  it('preserve shape info with inferred dim', async () => {
    const sparseTensor = sparseTensorValue2x3x4();
    const result =
        tf.sparse.sparseReshape(sparseTensor.ind, sparseTensor.shape, [2, -1]);
    expectArraysClose(await result.outputShape.data(), [2, 3 * 4]);
  });

  it('throw error if more than one inferred dim', async () => {
    const sparseTensor = sparseTensorValue2x3x4();
    expect(() => tf.sparse.sparseReshape(sparseTensor.ind, sparseTensor.shape, [
      -1, 2, -1
    ])).toThrowError(/only one output dimension may be -1/);
  });

  it('throw error if impossible new shape', async () => {
    const sparseTensor = sparseTensorValue2x3x4();
    expect(() => tf.sparse.sparseReshape(sparseTensor.ind, sparseTensor.shape, [
      -1, 7
    ])).toThrowError(/multiple of 7/);
  });
  it('same shape', async () => {
    const sparseTensor = sparseTensorValue5x6();
    const result =
        tf.sparse.sparseReshape(sparseTensor.ind, sparseTensor.shape, [5, 6]);
    expectArraysClose(
        await result.outputIndices.data(), await sparseTensor.ind.data());
    expectArraysClose(await result.outputShape.data(), sparseTensor.shape);
  });

  it('same shape with inferred dim', async () => {
    const sparseTensor = sparseTensorValue5x6();
    const result =
        tf.sparse.sparseReshape(sparseTensor.ind, sparseTensor.shape, [-1, 6]);
    expectArraysClose(
        await result.outputIndices.data(), await sparseTensor.ind.data());
    expectArraysClose(await result.outputShape.data(), sparseTensor.shape);
  });

  it('new shape with same rank', async () => {
    const sparseTensor = sparseTensorValue5x6();
    const result =
        tf.sparse.sparseReshape(sparseTensor.ind, sparseTensor.shape, [3, 10]);
    expectArraysClose(
        await result.outputIndices.data(),
        [[0, 0], [0, 6], [0, 9], [1, 0], [2, 0], [2, 1]]);
    expectArraysClose(await result.outputShape.data(), [3, 10]);
  });

  it('new shape with same rank with inferred dim', async () => {
    const sparseTensor = sparseTensorValue5x6();
    const result =
        tf.sparse.sparseReshape(sparseTensor.ind, sparseTensor.shape, [3, -1]);
    expectArraysClose(
        await result.outputIndices.data(),
        [[0, 0], [0, 6], [0, 9], [1, 0], [2, 0], [2, 1]]);
    expectArraysClose(await result.outputShape.data(), [3, 10]);
  });
  it('up rank', async () => {
    const sparseTensor = sparseTensorValue5x6();
    const result = tf.sparse.sparseReshape(
        sparseTensor.ind, sparseTensor.shape, [2, 3, 5]);
    expectArraysClose(
        await result.outputIndices.data(),
        [[0, 0, 0], [0, 1, 1], [0, 1, 4], [0, 2, 0], [1, 1, 0], [1, 1, 1]]);
    expectArraysClose(await result.outputShape.data(), [2, 3, 5]);
  });
  it('up rank with inferred dim', async () => {
    const sparseTensor = sparseTensorValue5x6();
    const result = tf.sparse.sparseReshape(
        sparseTensor.ind, sparseTensor.shape, [2, -1, 5]);
    expectArraysClose(
        await result.outputIndices.data(),
        [[0, 0, 0], [0, 1, 1], [0, 1, 4], [0, 2, 0], [1, 1, 0], [1, 1, 1]]);
    expectArraysClose(await result.outputShape.data(), [2, 3, 5]);
  });

  it('down rank', async () => {
    const sparseTensor = sparseTensorValue2x3x4();
    const result =
        tf.sparse.sparseReshape(sparseTensor.ind, sparseTensor.shape, [6, 4]);
    expectArraysClose(
        await result.outputIndices.data(),
        [[0, 1], [1, 0], [1, 2], [3, 3], [4, 1], [4, 3], [5, 2]]);
    expectArraysClose(await result.outputShape.data(), [6, 4]);
  });

  it('down rank with inferred dim', async () => {
    const sparseTensor = sparseTensorValue2x3x4();
    const result =
        tf.sparse.sparseReshape(sparseTensor.ind, sparseTensor.shape, [6, -1]);
    expectArraysClose(
        await result.outputIndices.data(),
        [[0, 1], [1, 0], [1, 2], [3, 3], [4, 1], [4, 3], [5, 2]]);
    expectArraysClose(await result.outputShape.data(), [6, 4]);
  });

  it('throw error if mismatch size', async () => {
    const sparseTensor = sparseTensorValue5x6();
    expect(() => tf.sparse.sparseReshape(sparseTensor.ind, sparseTensor.shape, [
      4, 7
    ])).toThrowError(/Input to reshape is a tensor with 30 dense values/);
  });
});
