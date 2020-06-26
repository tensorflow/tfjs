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
import {Tensor2D} from '../tensor';
import {expectArraysClose} from '../test_util';
import {Rank} from '../types';

describeWithFlags('lstm', ALL_ENVS, () => {
  it('MultiRNNCell with 2 BasicLSTMCells', async () => {
    const lstmKernel1 = tf.tensor2d(
        [
          0.26242125034332275, -0.8787832260131836, 0.781475305557251,
          1.337337851524353, 0.6180247068405151, -0.2760246992111206,
          -0.11299663782119751, -0.46332040429115295, -0.1765323281288147,
          0.6807947158813477, -0.8326982855796814, 0.6732975244522095
        ],
        [3, 4]);
    const lstmBias1 = tf.tensor1d(
        [1.090713620185852, -0.8282332420349121, 0, 1.0889357328414917]);
    const lstmKernel2 = tf.tensor2d(
        [
          -1.893059492111206, -1.0185645818710327, -0.6270437240600586,
          -2.1829540729522705, -0.4583775997161865, -0.5454602241516113,
          -0.3114445209503174, 0.8450229167938232
        ],
        [2, 4]);
    const lstmBias2 = tf.tensor1d(
        [0.9906240105628967, 0.6248329877853394, 0, 1.0224634408950806]);

    const forgetBias = tf.scalar(1.0);
    const lstm1 = (data: Tensor2D, c: Tensor2D, h: Tensor2D) =>
        tf.basicLSTMCell(forgetBias, lstmKernel1, lstmBias1, data, c, h);
    const lstm2 = (data: Tensor2D, c: Tensor2D, h: Tensor2D) =>
        tf.basicLSTMCell(forgetBias, lstmKernel2, lstmBias2, data, c, h);
    const c = [
      tf.zeros<Rank.R2>([1, lstmBias1.shape[0] / 4]),
      tf.zeros<Rank.R2>([1, lstmBias2.shape[0] / 4])
    ];
    const h = [
      tf.zeros<Rank.R2>([1, lstmBias1.shape[0] / 4]),
      tf.zeros<Rank.R2>([1, lstmBias2.shape[0] / 4])
    ];

    const onehot = tf.buffer<Rank.R2>([1, 2], 'float32');
    onehot.set(1.0, 0, 0);

    const output = tf.multiRNNCell([lstm1, lstm2], onehot.toTensor(), c, h);

    expectArraysClose(await output[0][0].data(), [-0.7440074682235718]);
    expectArraysClose(await output[0][1].data(), [0.7460772395133972]);
    expectArraysClose(await output[1][0].data(), [-0.5802832245826721]);
    expectArraysClose(await output[1][1].data(), [0.5745711922645569]);
  });
});

describeWithFlags('multiRNN throws when passed non-tensor', ALL_ENVS, () => {
  it('input: data', () => {
    const lstmKernel1: tf.Tensor2D = tf.zeros([3, 4]);
    const lstmBias1: tf.Tensor1D = tf.zeros([4]);
    const lstmKernel2: tf.Tensor2D = tf.zeros([2, 4]);
    const lstmBias2: tf.Tensor1D = tf.zeros([4]);

    const forgetBias = tf.scalar(1.0);
    const lstm1 = (data: Tensor2D, c: Tensor2D, h: Tensor2D) =>
        tf.basicLSTMCell(forgetBias, lstmKernel1, lstmBias1, data, c, h);
    const lstm2 = (data: Tensor2D, c: Tensor2D, h: Tensor2D) =>
        tf.basicLSTMCell(forgetBias, lstmKernel2, lstmBias2, data, c, h);
    const c = [
      tf.zeros<Rank.R2>([1, lstmBias1.shape[0] / 4]),
      tf.zeros<Rank.R2>([1, lstmBias2.shape[0] / 4])
    ];
    const h = [
      tf.zeros<Rank.R2>([1, lstmBias1.shape[0] / 4]),
      tf.zeros<Rank.R2>([1, lstmBias2.shape[0] / 4])
    ];

    expect(() => tf.multiRNNCell([lstm1, lstm2], {} as tf.Tensor2D, c, h))
        .toThrowError(
            /Argument 'data' passed to 'multiRNNCell' must be a Tensor/);
  });

  it('input: c', () => {
    const lstmKernel1: tf.Tensor2D = tf.zeros([3, 4]);
    const lstmBias1: tf.Tensor1D = tf.zeros([4]);
    const lstmKernel2: tf.Tensor2D = tf.zeros([2, 4]);
    const lstmBias2: tf.Tensor1D = tf.zeros([4]);

    const forgetBias = tf.scalar(1.0);
    const lstm1 = (data: Tensor2D, c: Tensor2D, h: Tensor2D) =>
        tf.basicLSTMCell(forgetBias, lstmKernel1, lstmBias1, data, c, h);
    const lstm2 = (data: Tensor2D, c: Tensor2D, h: Tensor2D) =>
        tf.basicLSTMCell(forgetBias, lstmKernel2, lstmBias2, data, c, h);

    const h = [
      tf.zeros<Rank.R2>([1, lstmBias1.shape[0] / 4]),
      tf.zeros<Rank.R2>([1, lstmBias2.shape[0] / 4])
    ];
    const data: tf.Tensor2D = tf.zeros([1, 2]);

    expect(() => tf.multiRNNCell([lstm1, lstm2], data, [{} as tf.Tensor2D], h))
        .toThrowError(
            /Argument 'c\[0\]' passed to 'multiRNNCell' must be a Tensor/);
  });

  it('input: h', () => {
    const lstmKernel1: tf.Tensor2D = tf.zeros([3, 4]);
    const lstmBias1: tf.Tensor1D = tf.zeros([4]);
    const lstmKernel2: tf.Tensor2D = tf.zeros([2, 4]);
    const lstmBias2: tf.Tensor1D = tf.zeros([4]);

    const forgetBias = tf.scalar(1.0);
    const lstm1 = (data: Tensor2D, c: Tensor2D, h: Tensor2D) =>
        tf.basicLSTMCell(forgetBias, lstmKernel1, lstmBias1, data, c, h);
    const lstm2 = (data: Tensor2D, c: Tensor2D, h: Tensor2D) =>
        tf.basicLSTMCell(forgetBias, lstmKernel2, lstmBias2, data, c, h);
    const c = [
      tf.zeros<Rank.R2>([1, lstmBias1.shape[0] / 4]),
      tf.zeros<Rank.R2>([1, lstmBias2.shape[0] / 4])
    ];
    const data: tf.Tensor2D = tf.zeros([1, 2]);

    expect(() => tf.multiRNNCell([lstm1, lstm2], data, c, [{} as tf.Tensor2D]))
        .toThrowError(
            /Argument 'h\[0\]' passed to 'multiRNNCell' must be a Tensor/);
  });
});
