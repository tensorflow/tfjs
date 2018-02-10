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

import * as dl from '../index';
// tslint:disable-next-line:max-line-length
import {ALL_ENVS, describeWithFlags, expectArraysClose} from '../test_util';
import {Tensor2D} from './tensor';
import {Rank} from './types';

describeWithFlags('lstm', ALL_ENVS, () => {
  it('MultiRNNCell with 2 BasicLSTMCells', () => {
    const lstmKernel1 = dl.tensor2d(
        [
          0.26242125034332275, -0.8787832260131836, 0.781475305557251,
          1.337337851524353, 0.6180247068405151, -0.2760246992111206,
          -0.11299663782119751, -0.46332040429115295, -0.1765323281288147,
          0.6807947158813477, -0.8326982855796814, 0.6732975244522095
        ],
        [3, 4]);
    const lstmBias1 = dl.tensor1d(
        [1.090713620185852, -0.8282332420349121, 0, 1.0889357328414917]);
    const lstmKernel2 = dl.tensor2d(
        [
          -1.893059492111206, -1.0185645818710327, -0.6270437240600586,
          -2.1829540729522705, -0.4583775997161865, -0.5454602241516113,
          -0.3114445209503174, 0.8450229167938232
        ],
        [2, 4]);
    const lstmBias2 = dl.tensor1d(
        [0.9906240105628967, 0.6248329877853394, 0, 1.0224634408950806]);

    const forgetBias = dl.scalar(1.0);
    const lstm1 = (data: Tensor2D, c: Tensor2D, h: Tensor2D) =>
        dl.basicLSTMCell(forgetBias, lstmKernel1, lstmBias1, data, c, h);
    const lstm2 = (data: Tensor2D, c: Tensor2D, h: Tensor2D) =>
        dl.basicLSTMCell(forgetBias, lstmKernel2, lstmBias2, data, c, h);
    const c = [
      dl.zeros<Rank.R2>([1, lstmBias1.shape[0] / 4]),
      dl.zeros<Rank.R2>([1, lstmBias2.shape[0] / 4])
    ];
    const h = [
      dl.zeros<Rank.R2>([1, lstmBias1.shape[0] / 4]),
      dl.zeros<Rank.R2>([1, lstmBias2.shape[0] / 4])
    ];

    const onehot = dl.buffer<Rank.R2>([1, 2], 'float32');
    onehot.set(1.0, 0, 0);

    const output = dl.multiRNNCell([lstm1, lstm2], onehot.toTensor(), c, h);

    expectArraysClose(output[0][0], [-0.7440074682235718]);
    expectArraysClose(output[0][1], [0.7460772395133972]);
    expectArraysClose(output[1][0], [-0.5802832245826721]);
    expectArraysClose(output[1][1], [0.5745711922645569]);
  });

  it('basicLSTMCell with batch=2', () => {
    const lstmKernel = dl.randomNormal<Rank.R2>([3, 4]);
    const lstmBias = dl.randomNormal<Rank.R1>([4]);
    const forgetBias = dl.scalar(1.0);

    const data = dl.randomNormal<Rank.R2>([1, 2]);
    const batchedData = dl.concat2d(data, data, 0);  // 2x2
    const c = dl.randomNormal<Rank.R2>([1, 1]);
    const batchedC = dl.concat2d(c, c, 0);  // 2x1
    const h = dl.randomNormal<Rank.R2>([1, 1]);
    const batchedH = dl.concat2d(h, h, 0);  // 2x1
    const [newC, newH] = dl.basicLSTMCell(
        forgetBias, lstmKernel, lstmBias, batchedData, batchedC, batchedH);
    expect(newC.get(0, 0)).toEqual(newC.get(1, 0));
    expect(newH.get(0, 0)).toEqual(newH.get(1, 0));
  });
});
