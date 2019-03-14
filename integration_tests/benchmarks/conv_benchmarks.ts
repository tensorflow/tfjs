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

import {BenchmarkTest} from './types';
import * as benchmark_util from './util';

export interface ConvParams {
  inDepth: number;
  filterSize: number;
  stride: number;
  pad: 'valid'|'same'|number;
}

export interface RegularConvParams extends ConvParams {
  outDepth: number;
}

export interface DepthwiseConvParams extends ConvParams {
  channelMul: number;
}

export class ConvGPUBenchmark implements BenchmarkTest {
  async run(size: number, opType: string, params: ConvParams): Promise<number> {
    tf.setBackend('webgl');

    const inDepth = params.inDepth;
    const inShape: [number, number, number] = [size, size, inDepth];
    const filterSize = params.filterSize;
    const stride = params.stride;
    const pad = params.pad;

    let x: tf.Tensor3D = tf.randomUniform(inShape, -1, 1);
    let W: tf.Tensor4D;

    let benchmark: () => tf.Tensor;
    if (opType === 'regular') {
      const regParams = params as RegularConvParams;
      const wShape: [number, number, number, number] =
          [filterSize, filterSize, inDepth, regParams.outDepth];
      W = tf.randomUniform(wShape, -1, 1);
      benchmark = () => x.conv2d(W, stride, pad);
    } else if (opType === 'transposed') {
      const regParams = params as RegularConvParams;
      const wShape: [number, number, number, number] =
          [filterSize, filterSize, inDepth, regParams.outDepth];
      W = tf.randomUniform(wShape, -1, 1);
      x = tf.randomUniform([size, size, regParams.outDepth], -1, 1);

      benchmark = () =>
          x.conv2dTranspose(W, [size, size, inDepth], stride, pad);
    } else if (opType === 'depthwise') {
      const depthwiseParams = params as DepthwiseConvParams;
      const wShape: [number, number, number, number] =
          [filterSize, filterSize, inDepth, depthwiseParams.channelMul];
      W = tf.randomUniform(wShape, -1, 1);

      benchmark = () => x.depthwiseConv2D(W, stride, pad);
    } else {
      throw new Error(`Unknown option ${opType}`);
    }

    const time = await benchmark_util.warmupAndBenchmarkGPU(benchmark);

    x.dispose();
    W.dispose();

    return time;
  }
}
