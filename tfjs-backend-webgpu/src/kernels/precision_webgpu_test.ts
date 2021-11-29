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

import * as tf from '@tensorflow/tfjs-core';
import {test_util} from '@tensorflow/tfjs-core';
import {describeWebGPU} from '../test_util';

const expectArraysClose = test_util.expectArraysClose;

describeWebGPU('precision', () => {
  it('float precision', async () => {
    const batch = 262145;
    const size = batch * 4 * 4 * 4;

    const aData = new Float32Array(size);
    const bData = new Float32Array(size / 4);
    for (let i = 0; i < size; i++) {
      aData[i] = i - 100;
      if (i % 4 === 0) {
        const iB = i / 4;
        bData[iB] = iB - 100;
      }
    }

    const aTensor = tf.tensor4d(aData, [batch, 4, 4, 4]);
    const bTensor = tf.tensor4d(bData, [batch, 4, 4, 1]);

    const gpuData = await tf.add(aTensor, bTensor).data();
    const expected = [
      20971312, 20971312, 20971316, 20971316, 20971316, 20971318, 20971320,
      20971320, 20971322, 20971324, 20971324, 20971326, 20971328, 20971328,
      20971330, 20971332, 20971332, 20971332, 20971336, 20971336
    ];
    expectArraysClose(Array.from(gpuData).slice(16777210, 16777230), expected);
  });
});
