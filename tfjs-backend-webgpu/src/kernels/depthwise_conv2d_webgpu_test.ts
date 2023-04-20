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

import * as tf from '@tensorflow/tfjs-core';
import {test_util} from '@tensorflow/tfjs-core';
import {describeWebGPU} from '../test_util';

const expectArraysClose = test_util.expectArraysClose;

describeWebGPU('depthwise conv2d nchw', () => {
  it('input=1x1x3x3,f=2,s=1,d=1,p=valid,chMul=1', async () => {
    const fSize = 2;
    const pad = 'valid';
    const stride = 1;
    const chMul = 1;
    const inDepth = 1;

    const x = tf.tensor4d([0, 1, 2, 3, 4, 5, 6, 7, 8], [1, inDepth, 3, 3]);
    const w = tf.tensor4d(
        [2, 3, 4, 5],
        [fSize, fSize, inDepth, chMul],
    );

    const result = tf.depthwiseConv2d(x, w, stride, pad, 'NCHW');
    expect(result.shape).toEqual([1, 1, 2, 2]);
    const expected = [35, 49, 77, 91];
    const resValue = await result.data();
    expectArraysClose(resValue, expected);
  });

  it('input=1x1x3x3,f=2,s=1,d=2,p=valid,chMul=1', async () => {
    const fSize = 2;
    const pad = 'valid';
    const stride = 1;
    const chMul = 1;
    const inDepth = 2;

    const x = tf.tensor4d(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 8, 0, 1, 3, 5, 7],
        [1, inDepth, 3, 3]);
    const w = tf.tensor4d(
        [2, 6, 3, 7, 4, 8, 5, 9],
        [fSize, fSize, inDepth, chMul],
    );

    const result = tf.depthwiseConv2d(x, w, stride, pad, 'NCHW');
    expect(result.shape).toEqual([1, 2, 2, 2]);
    const expected = [35, 49, 77, 91, 104, 75, 117, 110];
    const resValue = await result.data();
    expectArraysClose(resValue, expected);
  });

  it('input=1x1x5x5,f=5,s=1,d=2,p=same,chMul=1', async () => {
    const fSize = 5;
    const pad = 'same';
    const stride = 1;
    const chMul = 1;
    const inDepth = 2;

    const x = tf.tensor4d(
        [
          0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7,
          8, 0, 1, 2, 3, 4, 5, 6, 4, 5, 6, 7, 8, 0, 1, 2, 3,
          4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3
        ],
        [1, inDepth, 5, 5]);
    const w = tf.tensor4d(
        [
          3, 2, 1, 0, 3, 2, 1, 0, 4, 5, 3, 2, 1, 0, 3, 4, 2,
          1, 0, 5, 4, 3, 2, 1, 0, 0, 3, 4, 2, 1, 0, 5, 4, 3,
          2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 4, 5, 3, 2, 1
        ],
        [fSize, fSize, inDepth, chMul],
    );

    const result = tf.depthwiseConv2d(x, w, stride, pad, 'NCHW');
    expect(result.shape).toEqual([1, 2, 5, 5]);
    const resValue = await result.data();
    const expected = [
      47,  85, 91,  95,  54,  93,  122, 109, 101, 91,  99,  145, 203,
      166, 82, 105, 135, 144, 153, 121, 66,  95,  128, 95,  81,  69,
      75,  97, 107, 57,  109, 134, 202, 166, 107, 126, 119, 212, 148,
      116, 66, 104, 139, 85,  55,  52,  82,  122, 66,  47
    ];
    expectArraysClose(resValue, expected);
  });
});
