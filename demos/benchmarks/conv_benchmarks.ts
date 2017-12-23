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

// tslint:disable-next-line:max-line-length
import {Array1D, Array3D, Array4D, conv_util, ENV, NDArray} from 'deeplearn';

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
    const math = ENV.math;

    const inDepth = params.inDepth;
    const inShape: [number, number, number] = [size, size, inDepth];
    const filterSize = params.filterSize;
    const stride = params.stride;
    const pad = params.pad;

    let x = Array3D.randUniform(inShape, -1, 1);
    let W: Array4D;
    let b: Array1D;

    let benchmark: () => NDArray;
    if (opType === 'regular') {
      const regParams = params as RegularConvParams;
      const wShape = conv_util.computeWeightsShape4D(
          inDepth, regParams.outDepth, filterSize, filterSize);
      W = Array4D.randUniform(wShape, -1, 1);
      b = Array1D.randUniform([regParams.outDepth], -1, 1);

      benchmark = () => math.conv2d(x, W, b, stride, pad);
    } else if (opType === 'transposed') {
      const regParams = params as RegularConvParams;
      const wShape = conv_util.computeWeightsShape4D(
          inDepth, regParams.outDepth, filterSize, filterSize);
      W = Array4D.randUniform(wShape, -1, 1);
      x = Array3D.randUniform([size, size, regParams.outDepth], -1, 1);

      benchmark = () =>
          math.conv2dTranspose(x, W, [size, size, inDepth], stride, pad);
    } else if (opType === 'depthwise') {
      const depthwiseParams = params as DepthwiseConvParams;
      const wShape = conv_util.computeWeightsShape4D(
          inDepth, depthwiseParams.channelMul, filterSize, filterSize);
      W = Array4D.randUniform(wShape, -1, 1);

      benchmark = () => math.depthwiseConv2D(x, W, stride, pad);
    } else {
      throw new Error(`Unknown option ${opType}`);
    }

    const time = await benchmark_util.warmupAndBenchmarkGPU(math, benchmark);

    x.dispose();
    W.dispose();
    if (b != null) {
      b.dispose();
    }

    return time;
  }
}
