/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import * as test_util from '../test_util';
import {MathTests} from '../test_util';

import {Array1D, Array2D, Scalar} from './ndarray';

// math.basicLSTMCell
{
  const tests: MathTests = it => {
    it('MultiRNNCell with 2 BasicLSTMCells', math => {
      const lstmKernel1 = Array2D.new([3, 4], [
        0.26242125034332275, -0.8787832260131836, 0.781475305557251,
        1.337337851524353, 0.6180247068405151, -0.2760246992111206,
        -0.11299663782119751, -0.46332040429115295, -0.1765323281288147,
        0.6807947158813477, -0.8326982855796814, 0.6732975244522095
      ]);
      const lstmBias1 = Array1D.new(
          [1.090713620185852, -0.8282332420349121, 0, 1.0889357328414917]);
      const lstmKernel2 = Array2D.new([2, 4], [
        -1.893059492111206, -1.0185645818710327, -0.6270437240600586,
        -2.1829540729522705, -0.4583775997161865, -0.5454602241516113,
        -0.3114445209503174, 0.8450229167938232
      ]);
      const lstmBias2 = Array1D.new(
          [0.9906240105628967, 0.6248329877853394, 0, 1.0224634408950806]);

      const forgetBias = Scalar.new(1.0);
      const lstm1 =
          math.basicLSTMCell.bind(math, forgetBias, lstmKernel1, lstmBias1);
      const lstm2 =
          math.basicLSTMCell.bind(math, forgetBias, lstmKernel2, lstmBias2);

      const c = [
        Array2D.zeros([1, lstmBias1.shape[0] / 4]),
        Array2D.zeros([1, lstmBias2.shape[0] / 4])
      ];
      const h = [
        Array2D.zeros([1, lstmBias1.shape[0] / 4]),
        Array2D.zeros([1, lstmBias2.shape[0] / 4])
      ];

      const onehot = Array2D.zeros([1, 2]);
      onehot.set(1.0, 0, 0);

      const output = math.multiRNNCell([lstm1, lstm2], onehot, c, h);

      test_util.expectArraysClose(output[0][0], [-0.7440074682235718]);
      test_util.expectArraysClose(output[0][1], [0.7460772395133972]);
      test_util.expectArraysClose(output[1][0], [-0.5802832245826721]);
      test_util.expectArraysClose(output[1][1], [0.5745711922645569]);
    });

    it('basicLSTMCell with batch=2', math => {
      const lstmKernel = Array2D.randNormal([3, 4]);
      const lstmBias = Array1D.randNormal([4]);
      const forgetBias = Scalar.new(1.0);

      const data = Array2D.randNormal([1, 2]);
      const batchedData = math.concat2D(data, data, 0);  // 2x2
      const c = Array2D.randNormal([1, 1]);
      const batchedC = math.concat2D(c, c, 0);  // 2x1
      const h = Array2D.randNormal([1, 1]);
      const batchedH = math.concat2D(h, h, 0);  // 2x1
      const [newC, newH] = math.basicLSTMCell(
          forgetBias, lstmKernel, lstmBias, batchedData, batchedC, batchedH);
      expect(newC.get(0, 0)).toEqual(newC.get(1, 0));
      expect(newH.get(0, 0)).toEqual(newH.get(1, 0));
    });
  };

  test_util.describeMathCPU('basicLSTMCell', [tests]);
  test_util.describeMathGPU('basicLSTMCell', [tests], [
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2},
    {'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1}
  ]);
}
