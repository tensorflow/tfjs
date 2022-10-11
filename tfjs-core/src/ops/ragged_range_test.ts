/**
 * @license
 * Copyright 2022 Google LLC.
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

async function runRaggedGather(
    starts: tf.Tensor, limits: tf.Tensor, deltas: tf.Tensor) {
  const output = tf.raggedRange(starts, limits, deltas);

  expect(output.rtNestedSplits.dtype).toEqual('int32');
  expect(output.rtNestedSplits.shape.length).toEqual(1);

  expect(output.rtDenseValues.dtype).toEqual(starts.dtype);
  expect(output.rtDenseValues.shape.length).toEqual(1);

  return {
    rtNestedSplits: await output.rtNestedSplits.data(),
    rtDenseValues: await output.rtDenseValues.data(),
    tensors: Object.values(output)
  };
}

describeWithFlags('raggedRange ', ALL_ENVS, () => {
  it('IntValues', async () => {
    const result = await runRaggedGather(
        tf.tensor1d([0, 5, 8, 5], 'int32'), tf.tensor1d([8, 7, 8, 1], 'int32'),
        tf.tensor1d([2, 1, 1, -1], 'int32'));

    // Expected: [[0, 2, 4, 6], [5, 6], [], [5, 4, 3, 2]]
    expectArraysClose(result.rtNestedSplits, [0, 4, 6, 6, 10]);
    expectArraysClose(result.rtDenseValues, [0, 2, 4, 6, 5, 6, 5, 4, 3, 2]);
  });

  it('FloatValues', async () => {
    const result = await runRaggedGather(
        tf.tensor1d([0, 5, 8, 5], 'float32'),
        tf.tensor1d([8, 7, 8, 1], 'float32'),
        tf.tensor1d([2, 1, 1, -1], 'float32'));

    // Expected: [[0, 2, 4, 6], [5, 6], [], [5, 4, 3, 2]]
    expectArraysClose(result.rtNestedSplits, [0, 4, 6, 6, 10]);
    expectArraysClose(result.rtDenseValues, [0, 2, 4, 6, 5, 6, 5, 4, 3, 2]);
  });

  it('RangeSizeOverflow', async () => {
    await expectAsync(runRaggedGather(
                          tf.tensor1d([1.1, 0.1], 'float32'),
                          tf.tensor1d([10, 1e10], 'float32'),
                          tf.tensor1d([1, 1e-10], 'float32')))
        .toBeRejectedWithError(
            'Requires ((limit - start) / delta) <= 2147483647');
  });

  it('BroadcastDeltas', async () => {
    const result = await runRaggedGather(
        tf.tensor1d([0, 5, 8], 'int32'), tf.tensor1d([8, 7, 8], 'int32'),
        tf.scalar(1, 'int32'));

    // Expected: [[0, 1, 2, 3, 4, 5, 6, 7], [5, 6], []]
    expectArraysClose(result.rtNestedSplits, [0, 8, 10, 10]);
    expectArraysClose(result.rtDenseValues, [0, 1, 2, 3, 4, 5, 6, 7, 5, 6]);
  });

  it('BroadcastLimitsAndDeltas', async () => {
    const result = await runRaggedGather(
        tf.scalar(0, 'int32'), tf.tensor1d([3, 0, 2], 'int32'),
        tf.scalar(1, 'int32'));

    // Expected: [[0, 1, 2], [], [0, 1]]
    expectArraysClose(result.rtNestedSplits, [0, 3, 3, 5]);
    expectArraysClose(result.rtDenseValues, [0, 1, 2, 0, 1]);
  });

  it('BroadcastStartsAndLimits', async () => {
    const result = await runRaggedGather(
        tf.scalar(0, 'int32'), tf.scalar(12, 'int32'),
        tf.tensor1d([3, 4, 5], 'int32'));

    // Expected: [[0, 3, 6, 9], [0, 4, 8], [0, 5, 10]]
    expectArraysClose(result.rtNestedSplits, [0, 4, 7, 10]);
    expectArraysClose(result.rtDenseValues, [0, 3, 6, 9, 0, 4, 8, 0, 5, 10]);
  });

  it('AllScalarInputs', async () => {
    const result = await runRaggedGather(
        tf.scalar(0, 'int32'), tf.scalar(5, 'int32'), tf.scalar(1, 'int32'));

    // Expected: [[0, 1, 2, 3, 4]]
    expectArraysClose(result.rtNestedSplits, [0, 5]);
    expectArraysClose(result.rtDenseValues, [0, 1, 2, 3, 4]);
  });

  it('InvalidArgsStarts', async () => {
    await expectAsync(runRaggedGather(
                          tf.tensor2d([0, 5, 8, 5], [4, 1], 'int32'),
                          tf.tensor1d([8, 7, 8, 1], 'int32'),
                          tf.tensor1d([2, 1, 1, -1], 'int32')))
        .toBeRejectedWithError('starts must be a scalar or vector');
  });

  it('InvalidArgsLimits', async () => {
    await expectAsync(runRaggedGather(
                          tf.tensor1d([0, 5, 8, 5], 'int32'),
                          tf.tensor2d([8, 7, 8, 1], [4, 1], 'int32'),
                          tf.tensor1d([2, 1, 1, -1], 'int32')))
        .toBeRejectedWithError('limits must be a scalar or vector');
  });

  it('InvalidArgsDeltas', async () => {
    await expectAsync(runRaggedGather(
                          tf.tensor1d([0, 5, 8, 5], 'int32'),
                          tf.tensor1d([8, 7, 8, 1], 'int32'),
                          tf.tensor2d([2, 1, 1, -1], [4, 1], 'int32')))
        .toBeRejectedWithError('deltas must be a scalar or vector');
  });

  it('InvalidArgsShapeMismatch', async () => {
    await expectAsync(runRaggedGather(
                          tf.tensor1d([0, 5, 8, 5], 'int32'),
                          tf.tensor1d([7, 8, 1], 'int32'),
                          tf.tensor1d([2, 1, 1, -1], 'int32')))
        .toBeRejectedWithError(
            'starts, limits, and deltas must have the same shape');
  });

  it('InvalidArgsZeroDelta', async () => {
    await expectAsync(runRaggedGather(
                          tf.tensor1d([0, 5, 8, 5], 'int32'),
                          tf.tensor1d([7, 8, 8, 1], 'int32'),
                          tf.tensor1d([2, 1, 0, -1], 'int32')))
        .toBeRejectedWithError('Requires delta != 0');
  });

  it('EmptyRangePositiveDelta', async () => {
    const result = await runRaggedGather(
        tf.tensor1d([0, 5], 'int32'), tf.tensor1d([5, 0], 'int32'),
        tf.scalar(2, 'int32'));

    // Expected: [[0, 2, 4], []]
    expectArraysClose(result.rtNestedSplits, [0, 3, 3]);
    expectArraysClose(result.rtDenseValues, [0, 2, 4]);
  });

  it('EmptyRangeNegativeDelta', async () => {
    const result = await runRaggedGather(
        tf.tensor1d([0, 5], 'int32'), tf.tensor1d([5, 0], 'int32'),
        tf.scalar(-2, 'int32'));

    // Expected: [[], [5, 3, 1]]
    expectArraysClose(result.rtNestedSplits, [0, 0, 3]);
    expectArraysClose(result.rtDenseValues, [5, 3, 1]);
  });

  it('does not have memory leak.', async () => {
    const beforeDataIds = tf.engine().backend.numDataIds();

    const starts = tf.tensor1d([0, 5, 8, 5], 'int32');
    const limits = tf.tensor1d([8, 7, 8, 1], 'int32');
    const deltas = tf.tensor1d([2, 1, 1, -1], 'int32');
    const result = await runRaggedGather(starts, limits, deltas);

    const afterResDataIds = tf.engine().backend.numDataIds();
    expect(afterResDataIds).toEqual(beforeDataIds + 5);

    tf.dispose([starts, limits, deltas]);
    tf.dispose(result.tensors);

    const afterDisposeDataIds = tf.engine().backend.numDataIds();
    expect(afterDisposeDataIds).toEqual(beforeDataIds);
  });
});
