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

describeWithFlags('raggedTensorToTensor ', ALL_ENVS, () => {
  it('RaggedTensorToTensor', async () => {
    const shape = [4, 4];
    const values = [.1, .2, .3, .4, .5, .6, .7, .8, .9];
    const defaultValue = 1.5;
    const rowPartitionTensors = [
      tf.scalar(4, 'int32'), tf.tensor1d([0, 0, 0, 2, 2, 2, 2, 3, 3], 'int32')
    ];
    const rowPartitionTypes = ['FIRST_DIM_SIZE', 'VALUE_ROWIDS'];
    const result = tf.raggedTensorToTensor(
        shape, values, defaultValue, rowPartitionTensors, rowPartitionTypes);

    expectArraysEqual(result.shape, shape);
    expectArraysClose(await result.data(), [
      .1, .2, .3, 1.5, 1.5, 1.5, 1.5, 1.5, .4, .5, .6, .7, .8, .9, 1.5, 1.5
    ]);
  });

  it('RaggedTensorToTensorRowSplits', async () => {
    const shape = [4, 4];
    const values = [.1, .2, .3, .4, .5, .6, .7, .8, .9];
    const defaultValue = 1.5;
    const rowPartitionTensors = [tf.tensor1d([0, 3, 3, 7, 9], 'int32')];
    const rowPartitionTypes = ['ROW_SPLITS'];
    const result = tf.raggedTensorToTensor(
        shape, values, defaultValue, rowPartitionTensors, rowPartitionTypes);

    expectArraysEqual(result.shape, shape);
    expectArraysClose(await result.data(), [
      .1, .2, .3, 1.5, 1.5, 1.5, 1.5, 1.5, .4, .5, .6, .7, .8, .9, 1.5, 1.5
    ]);
  });

  it('RaggedTensorToTensor3DParams', async () => {
    const shape = [5, 2, 3];
    const values = [.1, .2, .3, .4, .5, .6, .7, .8, .9];
    const defaultValue = 1.5;
    const rowPartitionTensors = [
      tf.scalar(5, 'int32'), tf.tensor1d([0, 1, 1, 3, 3, 4], 'int32'),
      tf.tensor1d([1, 1, 2, 3, 3, 4, 4, 4, 5], 'int32')
    ];
    const rowPartitionTypes =
        ['FIRST_DIM_SIZE', 'VALUE_ROWIDS', 'VALUE_ROWIDS'];
    const result = tf.raggedTensorToTensor(
        shape, values, defaultValue, rowPartitionTensors, rowPartitionTypes);

    expectArraysEqual(result.shape, shape);
    expectArraysClose(await result.data(), [
      1.5, 1.5, 1.5, 1.5, 1.5, 1.5, .1, .2, 1.5, .3, 1.5, 1.5, 1.5, 1.5, 1.5,
      1.5, 1.5, 1.5, .4,  .5,  1.5, .6, .7, .8,  .9, 1.5, 1.5, 1.5, 1.5, 1.5
    ]);
  });

  it('RaggedTensorToTensor3DParamsRowSplits', async () => {
    const shape = [5, 2, 3];
    const values = [.1, .2, .3, .4, .5, .6, .7, .8, .9];
    const defaultValue = 1.5;
    const rowPartitionTensors = [
      tf.tensor1d([0, 1, 3, 3, 5, 6], 'int32'),
      tf.tensor1d([0, 0, 2, 3, 5, 8, 9], 'int32')
    ];
    const rowPartitionTypes = ['ROW_SPLITS', 'ROW_SPLITS'];
    const result = tf.raggedTensorToTensor(
        shape, values, defaultValue, rowPartitionTensors, rowPartitionTypes);

    expectArraysEqual(result.shape, shape);
    expectArraysClose(await result.data(), [
      1.5, 1.5, 1.5, 1.5, 1.5, 1.5, .1, .2, 1.5, .3, 1.5, 1.5, 1.5, 1.5, 1.5,
      1.5, 1.5, 1.5, .4,  .5,  1.5, .6, .7, .8,  .9, 1.5, 1.5, 1.5, 1.5, 1.5
    ]);
  });

  it('RaggedTensorToTensor3DParamsRowSplits2', async () => {
    const shape = [3, 2, 3];
    const values = [0, 1, 2, 3];
    const defaultValue = 5;
    const rowPartitionTensors = [
      tf.tensor1d([0, 2, 2, 3], 'int32'), tf.tensor1d([0, 3, 3, 4], 'int32')
    ];
    const rowPartitionTypes = ['ROW_SPLITS', 'ROW_SPLITS'];
    const result = tf.raggedTensorToTensor(
        shape, values, defaultValue, rowPartitionTensors, rowPartitionTypes);

    expectArraysEqual(result.shape, shape);
    expectArraysClose(
        await result.data(),
        [0, 1, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 5, 5, 5, 5, 5]);
  });

  it('RaggedTensorToTensor4DParams', async () => {
    const shape = [4, 2, 3, 2];
    const values = [1, 2, 3, 4, 5, 6, 7, 8];
    const defaultValue = 15;
    const rowPartitionTensors = [
      tf.scalar(5, 'int32'), tf.tensor1d([0, 1, 1], 'int32'),
      tf.tensor1d([1, 1, 1, 2], 'int32'),
      tf.tensor1d([0, 0, 1, 1, 2, 2, 3, 3], 'int32')
    ];
    const rowPartitionTypes =
        ['FIRST_DIM_SIZE', 'VALUE_ROWIDS', 'VALUE_ROWIDS', 'VALUE_ROWIDS'];
    const result = tf.raggedTensorToTensor(
        shape, values, defaultValue, rowPartitionTensors, rowPartitionTypes);

    expectArraysEqual(result.shape, shape);
    expectArraysClose(await result.data(), [
      15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 1,  2,  3,  4,
      5,  6,  7,  8,  15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
      15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15
    ]);
  });

  it('RaggedTensorToTensor4DParamsRowSplit', async () => {
    const shape = [4, 2, 3, 2];
    const values = [1, 2, 3, 4, 5, 6, 7, 8];
    const defaultValue = 15;
    const rowPartitionTensors = [
      tf.tensor1d([0, 1, 3], 'int32'), tf.tensor1d([0, 0, 3, 4], 'int32'),
      tf.tensor1d([0, 2, 4, 6, 8], 'int32')
    ];
    const rowPartitionTypes = ['ROW_SPLITS', 'ROW_SPLITS', 'ROW_SPLITS'];
    const result = tf.raggedTensorToTensor(
        shape, values, defaultValue, rowPartitionTensors, rowPartitionTypes);

    expectArraysEqual(result.shape, shape);
    expectArraysClose(await result.data(), [
      15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 1,  2,  3,  4,
      5,  6,  7,  8,  15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
      15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15
    ]);
  });

  it('RaggedTensorToTensorContractExpanded', async () => {
    const shape = [3, 5];
    const values = [.1, .2, .3, .4, .5, .6, .7, .8, .9];
    const defaultValue = 1.5;
    const rowPartitionTensors = [
      tf.scalar(4, 'int32'),
      tf.tensor1d([0, 0, 0, 2, 2, 2, 2, 3, 3], 'int32'),
    ];
    const rowPartitionTypes = ['FIRST_DIM_SIZE', 'VALUE_ROWIDS'];
    const result = tf.raggedTensorToTensor(
        shape, values, defaultValue, rowPartitionTensors, rowPartitionTypes);

    expectArraysEqual(result.shape, shape);
    expectArraysClose(await result.data(), [
      .1, .2, .3, 1.5, 1.5,     //
      1.5, 1.5, 1.5, 1.5, 1.5,  //
      .4, .5, .6, .7, 1.5
    ]);
  });

  it('RaggedTensorToTensorContractExpandedDense', async () => {
    const shape = [3, 5, 2];
    const values = tf.tensor2d(
        [
          .1, 1.1, .2, 1.2, .3, 1.3, .4, 1.4, .5, 1.5, .6, 1.6, .7, 1.7, .8,
          1.8, .9, 1.9
        ],
        [9, 2]);
    const defaultValue = 1.5;
    const rowPartitionTensors = [
      tf.scalar(4, 'int32'),
      tf.tensor1d([0, 0, 0, 2, 2, 2, 2, 3, 3], 'int32'),
    ];
    const rowPartitionTypes = ['FIRST_DIM_SIZE', 'VALUE_ROWIDS'];
    const result = tf.raggedTensorToTensor(
        shape, values, defaultValue, rowPartitionTensors, rowPartitionTypes);

    expectArraysEqual(result.shape, shape);
    expectArraysClose(await result.data(), [
      .1,  1.1, .2,  1.2, .3,  1.3, 1.5, 1.5, 1.5, 1.5,  //
      1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,  //
      .4,  1.4, .5,  1.5, .6,  1.6, .7,  1.7, 1.5, 1.5
    ]);
  });

  it('RaggedTensorToTensorConstrained', async () => {
    const shape = [3, 3];
    const values = [.1, .2, .3, .4, .5, .6, .7, .8, .9];
    const defaultValue = 1.5;
    const rowPartitionTensors = [
      tf.scalar(4, 'int32'),
      tf.tensor1d([0, 0, 0, 2, 2, 2, 2, 3, 3], 'int32'),
    ];
    const rowPartitionTypes = ['FIRST_DIM_SIZE', 'VALUE_ROWIDS'];
    const result = tf.raggedTensorToTensor(
        shape, values, defaultValue, rowPartitionTensors, rowPartitionTypes);

    expectArraysEqual(result.shape, shape);
    expectArraysClose(await result.data(), [
      .1, .2, .3,     //
      1.5, 1.5, 1.5,  //
      .4, .5, .6
    ]);
  });

  it('RaggedTensorToTensor3DParamsConstrained', async () => {
    const shape = [4, 1, 2];
    const values = [.1, .2, .3, .4, .5, .6, .7, .8, .9];
    const defaultValue = 1.5;
    const rowPartitionTensors = [
      tf.scalar(5, 'int32'),
      tf.tensor1d([0, 1, 1, 3, 3, 4], 'int32'),
      tf.tensor1d([1, 1, 2, 3, 3, 4, 4, 4, 5], 'int32'),
    ];
    const rowPartitionTypes =
        ['FIRST_DIM_SIZE', 'VALUE_ROWIDS', 'VALUE_ROWIDS'];
    const result = tf.raggedTensorToTensor(
        shape, values, defaultValue, rowPartitionTensors, rowPartitionTypes);

    expectArraysEqual(result.shape, shape);
    expectArraysClose(
        await result.data(), [1.5, 1.5, .1, .2, 1.5, 1.5, .4, .5]);
  });

  it('RaggedTensorToTensor4DParamsConstrained', async () => {
    const shape = [2, 2, 2, 2];
    const values = [1, 2, 3, 4, 5, 6, 7, 8];
    const defaultValue = 15;
    const rowPartitionTensors = [
      tf.scalar(5, 'int32'),
      tf.tensor1d([0, 1, 1], 'int32'),
      tf.tensor1d([1, 1, 1, 2], 'int32'),
      tf.tensor1d([0, 0, 1, 1, 2, 2, 3, 3], 'int32'),
    ];
    const rowPartitionTypes =
        ['FIRST_DIM_SIZE', 'VALUE_ROWIDS', 'VALUE_ROWIDS', 'VALUE_ROWIDS'];
    const result = tf.raggedTensorToTensor(
        shape, values, defaultValue, rowPartitionTensors, rowPartitionTypes);

    expectArraysEqual(result.shape, shape);
    expectArraysClose(await result.data(), [
      15, 15, 15, 15,  //
      15, 15, 15, 15,  //
      1, 2, 3, 4,      //
      7, 8, 15, 15,    //
    ]);
  });

  it('shape wrong dimensions', async () => {
    const shape = [10, 7, 10, 20];
    const values = [1, 2, 3, 4];
    const defaultValue = 15;
    const rowPartitionTensors = [
      tf.scalar(5, 'int32'),
      tf.tensor1d([0, 1, 1], 'int32'),
      tf.tensor1d([1, 1, 1, 2], 'int32'),
    ];
    const rowPartitionTypes =
        ['FIRST_DIM_SIZE', 'VALUE_ROWIDS', 'VALUE_ROWIDS'];

    expect(
        () => tf.raggedTensorToTensor(
            shape, values, defaultValue, rowPartitionTensors,
            rowPartitionTypes))
        .toThrowError(/are incompatible/);
  });

  it('does not have memory leak.', async () => {
    const beforeDataIds = tf.engine().backend.numDataIds();

    const rowPartitionTensors = [
      tf.scalar(4, 'int32'), tf.tensor1d([0, 0, 0, 2, 2, 2, 2, 3, 3], 'int32')
    ];
    const result = tf.raggedTensorToTensor(
        [4, 4], [.1, .2, .3, .4, .5, .6, .7, .8, .9], 1.5, rowPartitionTensors,
        ['FIRST_DIM_SIZE', 'VALUE_ROWIDS']);

    await result.data();

    const afterResDataIds = tf.engine().backend.numDataIds();
    expect(afterResDataIds).toEqual(beforeDataIds + 3);

    result.dispose();
    rowPartitionTensors.map(tensor => tensor.dispose());

    const afterDisposeDataIds = tf.engine().backend.numDataIds();
    expect(afterDisposeDataIds).toEqual(beforeDataIds);
  });
});
