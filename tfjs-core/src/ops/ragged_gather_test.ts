/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
import {expectArraysClose, expectArraysEqual} from '../test_util';

async function runRaggedGather(
    indicesShape: number[], indices: number[], paramsNestedSplits: number[][],
    paramsDenseValuesShape: number[], paramsDenseValues: number[]) {
  const paramsRaggedRank = paramsNestedSplits.length;
  const numSplits = paramsRaggedRank + indicesShape.length - 1;

  const paramsNestedSplitsTensors =
      paramsNestedSplits.map(values => tf.tensor1d(values, 'int32'));
  const paramsDenseValuesTensor =
      tf.tensor(paramsDenseValues, paramsDenseValuesShape);
  const indicesTensor = tf.tensor(indices, indicesShape, 'int32');

  const output = tf.raggedGather(
      paramsNestedSplitsTensors, paramsDenseValuesTensor, indicesTensor,
      numSplits);

  tf.dispose(paramsNestedSplitsTensors);
  tf.dispose([paramsDenseValuesTensor, indicesTensor]);

  expect(output.outputDenseValues.dtype).toEqual('float32');

  output.outputNestedSplits.forEach(splits => {
    expect(splits.dtype).toEqual('int32');
    expect(splits.shape.length).toEqual(1);
  });

  return {
    outputDenseValues: await output.outputDenseValues.data(),
    outputDenseValuesShape: output.outputDenseValues.shape,
    outputNestedSplits: await Promise.all(
        output.outputNestedSplits.map(splits => splits.data())),
    tensors: output.outputNestedSplits.concat([output.outputDenseValues])
  };
}

describeWithFlags('raggedGather ', ALL_ENVS, () => {
  it('RaggedGather', async () => {
    const result = await runRaggedGather(
        [4], [2, 1, 0, 3], [[0, 3, 3, 7, 9]], [9],
        [.1, .2, .3, .4, .5, .6, .7, .8, .9]);

    expect(result.outputNestedSplits.length).toEqual(1);
    expectArraysClose(result.outputNestedSplits[0], [0, 4, 4, 7, 9]);

    expectArraysClose(
        result.outputDenseValues, [.4, .5, .6, .7, .1, .2, .3, .8, .9]);
    expectArraysEqual(result.outputDenseValuesShape, [9]);
  });

  it('RaggedGather3DParams', async () => {
    const result = await runRaggedGather(
        [5], [2, 1, 0, 2, 3], [[0, 1, 3, 3, 5, 6], [0, 0, 2, 3, 5, 8, 9]], [9],
        [.1, .2, .3, .4, .5, .6, .7, .8, .9]);

    expect(result.outputNestedSplits.length).toEqual(2);
    expectArraysClose(result.outputNestedSplits[0], [0, 0, 2, 3, 3, 5]);
    expectArraysClose(result.outputNestedSplits[1], [0, 2, 3, 3, 5, 8]);

    expectArraysClose(
        result.outputDenseValues, [.1, .2, .3, .4, .5, .6, .7, .8]);
    expectArraysEqual(result.outputDenseValuesShape, [8]);
  });

  it('RaggedGather4DParams', async () => {
    const result = await runRaggedGather(
        [4], [2, 1, 0, 2], [[0, 1, 3, 3], [0, 0, 3, 4]], [4, 2],
        [1, 2, 3, 4, 5, 6, 7, 8]);

    expect(result.outputNestedSplits.length).toEqual(2);
    expectArraysClose(result.outputNestedSplits[0], [0, 0, 2, 3, 3]);
    expectArraysClose(result.outputNestedSplits[1], [0, 3, 4, 4]);

    expectArraysClose(result.outputDenseValues, [1, 2, 3, 4, 5, 6, 7, 8]);
    expectArraysEqual(result.outputDenseValuesShape, [4, 2]);
  });

  it('RaggedGather2DIndices', async () => {
    const result = await runRaggedGather(
        [2, 2], [2, 1, 0, 3], [[0, 3, 3, 7, 9]], [9],
        [.1, .2, .3, .4, .5, .6, .7, .8, .9]);

    expect(result.outputNestedSplits.length).toEqual(2);
    expectArraysClose(result.outputNestedSplits[0], [0, 2, 4]);
    expectArraysClose(result.outputNestedSplits[1], [0, 4, 4, 7, 9]);

    expectArraysClose(
        result.outputDenseValues, [.4, .5, .6, .7, .1, .2, .3, .8, .9]);
    expectArraysEqual(result.outputDenseValuesShape, [9]);
  });

  it('RaggedGatherScalarIndices', async () => {
    const result = await runRaggedGather(
        [], [2], [[0, 3, 3, 7, 9]], [9], [.1, .2, .3, .4, .5, .6, .7, .8, .9]);

    expect(result.outputNestedSplits.length).toEqual(0);

    expectArraysClose(result.outputDenseValues, [.4, .5, .6, .7]);
    expectArraysEqual(result.outputDenseValuesShape, [4]);
  });

  it('OutOfBounds', async () => {
    await expectAsync(runRaggedGather([2], [2, 10], [[0, 3, 3, 7, 9]], [9], [
      .1, .2, .3, .4, .5, .6, .7, .8, .9
    ])).toBeRejectedWithError('indices[1] = 10 is not in [0, 4)');
  });

  it('InvalidSplitsNotSorted', async () => {
    await expectAsync(runRaggedGather(
                          [2], [0, 2], [[0, 3, 5, 2, 9]], [9],
                          [.1, .2, .3, .4, .5, .6, .7, .8, .9]))
        .toBeRejectedWithError(
            'Ragged splits must be sorted in ascending order');
  });

  it('InvalidSplitsNegative', async () => {
    await expectAsync(runRaggedGather([2], [0, 2], [[-1, 3, 2, 7, 9]], [9], [
      .1, .2, .3, .4, .5, .6, .7, .8, .9
    ])).toBeRejectedWithError('Ragged splits must be non-negative');
  });

  it('InvalidSplitsEmpty', async () => {
    await expectAsync(runRaggedGather([0], [], [[]], [0], []))
        .toBeRejectedWithError('Ragged splits may not be empty');
  });

  it('InvalidSplitsTooBig', async () => {
    await expectAsync(runRaggedGather(
                          [2], [0, 2], [[0, 20, 40, 80, 100]], [9],
                          [.1, .2, .3, .4, .5, .6, .7, .8, .9]))
        .toBeRejectedWithError('Ragged splits must not point past values');
  });

  it('BadValuesShape', async () => {
    await expectAsync(runRaggedGather([0], [], [[0]], [], [.1]))
        .toBeRejectedWithError('params.rank must be nonzero');
  });

  it('does not have memory leak.', async () => {
    const beforeDataIds = tf.engine().backend.numDataIds();

    const result = await runRaggedGather(
        [4], [2, 1, 0, 3], [[0, 3, 3, 7, 9]], [9],
        [.1, .2, .3, .4, .5, .6, .7, .8, .9]);

    const afterResDataIds = tf.engine().backend.numDataIds();
    expect(afterResDataIds).toEqual(beforeDataIds + 2);

    result.tensors.map(tensor => tensor.dispose());

    const afterDisposeDataIds = tf.engine().backend.numDataIds();
    expect(afterDisposeDataIds).toEqual(beforeDataIds);
  });
});
