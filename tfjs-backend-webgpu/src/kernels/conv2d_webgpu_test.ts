/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {describeWebGPU} from '../test_util';

function generateCaseInputs(totalSizeTensor: number, totalSizeFilter: number) {
  const inp = new Array(totalSizeTensor);
  const filt = new Array(totalSizeFilter);

  for (let i = 0; i < totalSizeTensor; i++) {
    inp[i] = i + 1;
  }
  for (let i = 0; i < totalSizeFilter; i++) {
    filt[i] = i + 1;
  }

  return {input: inp, filter: filt};
}

describeWebGPU('im2col as separate shader', () => {
  beforeAll(() => {
    tf.env().set('WEBGPU_CONV_SEPARATE_IM2COL_SHADER', true);
  });

  it('x=[1,4,4,1] f=[1,1,1,3] s=2 d=1 p=same', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [4, 4, inputDepth];
    const outputDepth = 3;
    const fSize = 1;
    const pad = 'same';
    const stride: [number, number] = [2, 2];

    const x = tf.tensor3d(
        [
          10, 30, 50, 70, 20, 40, 60, 80, -10, -30, -50, -70, -20, -40, -60, -80
        ],
        inputShape);
    const w = tf.tensor4d([1, 0.5, 1], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.conv2d(x, w, stride, pad);
    tf.test_util.expectArraysClose(
        await result.data(),
        [10, 5, 10, 50, 25, 50, -10, -5, -10, -50, -25, -50]);
  });

  it('x=[2,2,1] f=[1,1,1,2] s=1 d=1 p=0', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    const x = tf.tensor3d([1, 2, 3, 4], inputShape);
    const w = tf.tensor4d([2], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.conv2d(x, w, stride, pad);
    tf.test_util.expectArraysClose(await result.data(), [2, 4, 6, 8]);
  });

  it('x=[3,3,2] f=[2,2,2,1] s=1 d=1 p=valid', async () => {
    const pad = 'valid';
    const stride = 1;

    const x = tf.tensor3d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        [3, 3, 2]);
    const w = tf.tensor4d([.1, .2, .3, .4, .5, .6, .7, .8], [2, 2, 2, 1]);
    const result = tf.conv2d(x, w, stride, pad);

    const resultData = await result.data();
    expect(result.shape).toEqual([2, 2, 1]);
    tf.test_util.expectArraysClose(
        resultData, new Float32Array([25.6, 53.5, 157.0, 220.9]));
  });

  it('x=[4,2,1] f=[4,2,1,1] s=1 d=1 p=same', async () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const pad = 'same';
    const stride = 1;
    const dataFormat = 'NHWC';
    const dilation = 1;

    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [4, 2, inputDepth]);
    const w =
        tf.tensor4d([3, 1, 5, 0, 2, 7, 8, 9], [4, 2, inputDepth, outputDepth]);

    const result = tf.conv2d(x, w, stride, pad, dataFormat, dilation);
    const resultData = await result.data();
    expect(result.shape).toEqual([4, 2, 1]);
    tf.test_util.expectArraysClose(
        resultData, [133, 66, 200, 102, 108, 58, 56, 58]);
  });

  it('x=[2,2,1] f=[2,2,1,1] s=1 d=1 p=same', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 2;
    const pad = 'same';
    const stride = 1;
    const dataFormat = 'NHWC';
    const dilation = 1;

    const x = tf.tensor3d([1, 2, 3, 4], inputShape);
    const w =
        tf.tensor4d([3, 1, 5, 0], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.conv2d(x, w, stride, pad, dataFormat, dilation);

    const resultData = await result.data();
    expect(result.shape).toEqual([2, 2, 1]);
    tf.test_util.expectArraysClose(
        resultData, new Float32Array([20, 26, 13, 12]));
  });

  it('x=[2,2,1] f=[2,2,1,1] s=1 d=1 p=0', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 2;
    const pad = 0;
    const stride = 1;
    const dataFormat = 'NHWC';
    const dilation = 1;

    const x = tf.tensor3d([1, 2, 3, 4], inputShape);
    const w =
        tf.tensor4d([3, 1, 5, 0], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.conv2d(x, w, stride, pad, dataFormat, dilation);
    tf.test_util.expectArraysClose(await result.data(), [20]);
  });

  it('x=[1,3,6,1] f=[2,2,1,1] s=[1,2] d=1 p=valid', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number, number] = [1, 3, 6, inputDepth];
    const outputDepth = 1;
    const fSize = 2;
    const pad = 'valid';
    const stride: [number, number] = [1, 2];

    const inputs = generateCaseInputs(1 * 3 * 6 * inputDepth, fSize * fSize);
    const x = tf.tensor4d(inputs.input, inputShape);
    const w =
        tf.tensor4d(inputs.filter, [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.conv2d(x, w, stride, pad);
    tf.test_util.expectArraysClose(
        await result.data(), [58.0, 78.0, 98.0, 118.0, 138.0, 158.0]);
  });
});
