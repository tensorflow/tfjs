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

import * as dl from 'deeplearn';

import {BenchmarkTest} from './benchmark';
import * as benchmark_util from './benchmark_util';

export interface ConvParams {
  inDepth: number;
  filterSize: number;
  stride: number;
  pad: 'valid'|'same'|number;
}

export interface RegularConvParams extends ConvParams { outDepth: number; }

export interface DepthwiseConvParams extends ConvParams { channelMul: number; }

export class ConvGPUBenchmark implements BenchmarkTest {
  async run(size: number, opType: string, params: ConvParams): Promise<number> {
    const safeMode = false;
    const math = new dl.NDArrayMath('webgl', safeMode);
    dl.ENV.setMath(math);

    const inDepth = params.inDepth;
    const inShape: [number, number, number] = [size, size, inDepth, 3];
    const filterSize = params.filterSize;
    const stride = params.stride;
    const pad = params.pad;

    let x: dl.Tensor3D = dl.randomUniform(inShape, -1, 1);
    let W: dl.Tensor4D;
    let b: dl.Tensor1D;

    let benchmark: () => dl.Tensor;
    if (opType === 'regular') {
      const regParams = params as RegularConvParams;
      const wShape = dl.conv_util.computeWeightsShape4D(
          inDepth, regParams.outDepth, filterSize, filterSize);
      W = dl.randomUniform(wShape, -1, 1);
      b = dl.randomUniform([regParams.outDepth], -1, 1);

      benchmark = () => x.conv2d(W, b, stride, pad);
    } else if (opType === 'transposed') {
      const regParams = params as RegularConvParams;
      const wShape = dl.conv_util.computeWeightsShape4D(
          inDepth, regParams.outDepth, filterSize, filterSize);
      W = dl.randomUniform(wShape, -1, 1);
      x = dl.randomUniform([size, size, regParams.outDepth], -1, 1);

      benchmark = () =>
          x.conv2dTranspose(W, [size, size, inDepth], stride, pad);
    } else if (opType === 'depthwise') {
      const depthwiseParams = params as DepthwiseConvParams;
      const wShape = dl.conv_util.computeWeightsShape4D(
          inDepth, depthwiseParams.channelMul, filterSize, filterSize);
      W = dl.randomUniform(wShape, -1, 1);

      benchmark = () => x.depthwiseConv2D(W, stride, pad);
    } else {
      throw new Error(`Unknown option ${opType}`);
    }

    const time = await benchmark_util.warmupAndBenchmarkGPU(benchmark);

    x.dispose();
    W.dispose();
    math.dispose();
    if (b != null) {
      b.dispose();
    }

    return time;
  }
}
