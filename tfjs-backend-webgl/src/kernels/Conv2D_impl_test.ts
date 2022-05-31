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
import {backend_util} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import {describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {MathBackendWebGL} from '../backend_webgl';
import {WEBGL_ENVS} from '../backend_webgl_test_registry';

import {conv2dWithIm2Row, TimeObj} from './Conv2D_impl';


describeWithFlags('conv2dByMatMul', WEBGL_ENVS, () => {
  it('Benchmark x=[1,548, 548, 4] f=[9, 9, 4, 3]', async () => {
    const inputDepth = 4;
    const inShape: [number, number, number, number] = [1, 548, 548, inputDepth];
    const outputDepth = 3;
    const fSize = 9;
    const stride = 1;
    const dataFormat = 'NHWC';
    const dilation = 1;
    const pad = 'valid';

    const x = tf.randomUniform(inShape, -100, 100) as tf.Tensor4D;
    const w =
        tf.randomUniform([fSize, fSize, inputDepth, outputDepth], -100, 100) as
        tf.Tensor4D;

    const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
    const convInfo = backend_util.computeConv2DInfo(
        x.shape as [number, number, number, number],
        w.shape as [number, number, number, number], stride, dilation, pad,
        undefined /* roundingMode */, false /* depthwise */, $dataFormat);

    // Warm-up run
    const result = conv2dWithIm2Row(
        {x, filter: w, convInfo, backend: tf.backend() as MathBackendWebGL});
    expect(result.shape).toEqual([1, 540, 540, 3]);

    let averageImg2col = 0;
    let averageMatMul = 0;
    let averagePercentOfImg2col = 0;
    let averagePercentOfMatMul = 0;
    const numRounds = 100;
    for (let round = 0; round < numRounds; round++) {
      const x = tf.randomUniform(inShape, -100, 100) as tf.Tensor4D;
      const w = tf.randomUniform(
                    [fSize, fSize, inputDepth, outputDepth], -100, 100) as
          tf.Tensor4D;
      const time = {} as TimeObj;
      conv2dWithIm2Row({
        x,
        filter: w,
        convInfo,
        backend: tf.backend() as MathBackendWebGL,
        time
      });

      averageImg2col += time.img2colTime;
      averageMatMul += time.matmulTime;
      averagePercentOfImg2col += time.img2colTime / time.totalTime / numRounds;
      averagePercentOfMatMul += time.matmulTime / time.totalTime / numRounds;
    }

    averageImg2col = averageImg2col / numRounds;
    averageMatMul = averageMatMul / numRounds;

    console.log(`
    average time for Img2col: ${averageImg2col},
    average time for MatMul: ${averageMatMul},
    average percent for Img2col: ${averagePercentOfImg2col},
    average percent for MatMul: ${averagePercentOfMatMul}.
    `);
  });
});
