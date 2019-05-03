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
import {expectArraysClose} from '@tensorflow/tfjs-core/dist/test_util';

import * as tfwebgpu from './index';

describe('WebGPU backend - convolution tests', () => {
  beforeAll(() => tfwebgpu.ready);

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

    const resultData = await result.data();
    expect(result.shape).toEqual([2, 2, 3]);
    expectArraysClose(
        resultData,
        new Float32Array([10, 5, 10, 50, 25, 50, -10, -5, -10, -50, -25, -50]));
  });

  it('x=[3,3,2] f=[2,2,2,1] s=1 d=1 p=valid', async () => {
    const pad = 'valid';
    const stride = 1;

    const x = tf.tensor3d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        [3, 3, 2]);
    const w = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);

    const result = tf.conv2d(x, w, stride, pad);

    const resultData = await result.data();
    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(resultData, new Float32Array([
                        256,
                        535,
                        1570,
                        2209,
                      ]));
  });

  it('x=[1,3,6,1] f=[2,2,1,1] s=[1,2] d=1 p=valid', async () => {
    const inputDepth = 1;
    const inputShape: [number, number, number, number] = [1, 3, 6, inputDepth];
    const outputDepth = 1;
    const fSize = 2;
    const pad = 'valid';
    const stride: [number, number] = [1, 2];

    const input =
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];
    const filter = [1, 2, 3, 4];

    const x = tf.tensor4d(input, inputShape);
    const w = tf.tensor4d(filter, [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.conv2d(x, w, stride, pad);

    const resultData = await result.data();
    expect(result.shape).toEqual([1, 2, 3, 1]);
    expectArraysClose(
        resultData, new Float32Array([58, 78, 98, 118, 138, 158]));
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

    const resultData = await result.data();
    expect(result.shape).toEqual([1, 1, 1]);
    expectArraysClose(resultData, new Float32Array([20]));
  });

  it('x=[2,2,2,1] f=[1,1,1,1] s=1 d=1 p=0', async () => {
    const inputDepth = 1;
    const inShape: [number, number, number, number] = [2, 2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], inShape);
    const w = tf.tensor4d([2], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.conv2d(x, w, stride, pad);
    expect(result.shape).toEqual([2, 2, 2, 1]);
    const expected = [2, 4, 6, 8, 10, 12, 14, 16];

    const resultData = await result.data();
    expect(result.shape).toEqual([2, 2, 2, 1]);
    expectArraysClose(resultData, new Float32Array(expected));
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
    expectArraysClose(resultData, new Float32Array([20, 26, 13, 12]));
  });

  it('x=[2,4,1] f=[4,2,1,1] s=1 d=1 p=same', async () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const pad = 'same';
    const stride = 1;
    const dataFormat = 'NHWC';
    const dilation = 1;

    const x = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 4, inputDepth]);
    const w =
        tf.tensor4d([3, 1, 5, 0, 2, 7, 8, 9], [4, 2, inputDepth, outputDepth]);

    const result = tf.conv2d(x, w, stride, pad, dataFormat, dilation);

    const resultData = await result.data();
    expect(result.shape).toEqual([2, 4, 1]);
    expectArraysClose(
        resultData, new Float32Array([57, 71, 85, 36, 30, 39, 48, 52]));
  });
});
