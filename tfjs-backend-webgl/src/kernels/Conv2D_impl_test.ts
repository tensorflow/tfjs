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
import {backend_util, test_util} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {MathBackendWebGL} from '../backend_webgl';
import {WEBGL_ENVS} from '../backend_webgl_test_registry';

import {conv2dByMatMul, conv2dWithIm2Row} from './Conv2D_impl';

const {expectArraysClose} = test_util;

describeWithFlags('conv2dByMatMul', WEBGL_ENVS, () => {
  it('Should work for NCHW format when having bias', async () => {
    const inputDepth = 4;
    const inputShape: [number, number, number, number] = [1, inputDepth, 2, 2];
    const outputDepth = 4;
    const fSize = 1;
    const pad = 'same';
    const stride = 1;
    const dataFormat = 'NCHW';
    const dilation = 1;

    const x = tf.tensor4d(
        [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], inputShape);
    const w = tf.tensor4d(
        [3, 3, 3, 3, 1, 1, 1, 1, 5, 5, 5, 5, 0, 0, 0, 0],
        [fSize, fSize, inputDepth, outputDepth]);
    const bias = tf.tensor1d([1, 2, 1, 2]);

    const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
    const convInfo = backend_util.computeConv2DInfo(
        x.shape as [number, number, number, number],
        w.shape as [number, number, number, number], stride, dilation, pad,
        undefined /* roundingMode */, false /* depthwise */, $dataFormat);

    const result = conv2dByMatMul({
      x,
      filter: w,
      convInfo,
      backend: tf.backend() as MathBackendWebGL,
      bias
    });

    expect(result.shape).toEqual([1, 4, 2, 2]);
    expectArraysClose(
        tf.backend().readSync(result.dataId),
        [10, 19, 28, 37, 11, 20, 29, 38, 10, 19, 28, 37, 11, 20, 29, 38]);
  });

  it('Should work for NCHW format when having bias and multiple batches',
     async () => {
       const inputDepth = 4;
       const inputShape: [number, number, number, number] =
           [2, inputDepth, 2, 2];
       const outputDepth = 4;
       const fSize = 1;
       const pad = 'same';
       const stride = 1;
       const dataFormat = 'NCHW';
       const dilation = 1;

       const x = tf.tensor4d(
           [
             1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
             1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4
           ],
           inputShape);
       const w = tf.tensor4d(
           [3, 3, 3, 3, 1, 1, 1, 1, 5, 5, 5, 5, 0, 0, 0, 0],
           [fSize, fSize, inputDepth, outputDepth]);
       const bias = tf.tensor1d([1, 2, 1, 2]);

       const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
       const convInfo = backend_util.computeConv2DInfo(
           x.shape as [number, number, number, number],
           w.shape as [number, number, number, number], stride, dilation, pad,
           undefined /* roundingMode */, false /* depthwise */, $dataFormat);

       const result = conv2dByMatMul({
         x,
         filter: w,
         convInfo,
         backend: tf.backend() as MathBackendWebGL,
         bias
       });

       expect(result.shape).toEqual([2, 4, 2, 2]);
       expectArraysClose(tf.backend().readSync(result.dataId), [
         10, 19, 28, 37, 11, 20, 29, 38, 10, 19, 28, 37, 11, 20, 29, 38,
         10, 19, 28, 37, 11, 20, 29, 38, 10, 19, 28, 37, 11, 20, 29, 38
       ]);
     });
});

describeWithFlags('conv2dWithIm2Row', WEBGL_ENVS, () => {
  it('Should work when having bias', async () => {
    const inputDepth = 4;
    const inputShape: [number, number, number, number] = [1, inputDepth, 2, 2];
    const outputDepth = 4;
    const fSize = 2;
    const pad = 'same';
    const stride = 1;
    const dataFormat = 'NCHW';
    const dilation = 1;

    const x = tf.tensor4d(
        [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], inputShape);
    const w = tf.tensor4d(
        [
          3, 3, 3, 3, 1, 1, 1, 1, 5, 5, 5, 5, 0, 0, 0, 0, 3, 3, 3, 3, 1, 1,
          1, 1, 5, 5, 5, 5, 0, 0, 0, 0, 3, 3, 3, 3, 1, 1, 1, 1, 5, 5, 5, 5,
          0, 0, 0, 0, 3, 3, 3, 3, 1, 1, 1, 1, 5, 5, 5, 5, 0, 0, 0, 0,
        ],
        [fSize, fSize, inputDepth, outputDepth]);
    const bias = tf.tensor1d([1, 2, 1, 2]);

    const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
    const convInfo = backend_util.computeConv2DInfo(
        x.shape as [number, number, number, number],
        w.shape as [number, number, number, number], stride, dilation, pad,
        undefined /* roundingMode */, false /* depthwise */, $dataFormat);

    const result = conv2dWithIm2Row({
      x,
      filter: w,
      convInfo,
      backend: tf.backend() as MathBackendWebGL,
      bias
    });

    expect(result.shape).toEqual([1, 4, 2, 2]);
    expectArraysClose(
        tf.backend().readSync(result.dataId),
        [91, 55, 64, 37, 92, 56, 65, 38, 91, 55, 64, 37, 92, 56, 65, 38]);
  });
});
