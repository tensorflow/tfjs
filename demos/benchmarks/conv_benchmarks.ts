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
import {Array1D, Array3D, Array4D, conv_util, ENV, NDArray, NDArrayMathGPU} from 'deeplearn';

import {BenchmarkTest} from './benchmark';

export interface ConvBenchmarkParams {
  inDepth: number;
  outDepth: number;
  filterSize: number;
  stride: number;
}

export abstract class ConvBenchmark extends BenchmarkTest {
  constructor(protected params: ConvBenchmarkParams) {
    super(params);
  }
}

export class ConvGPUBenchmark extends ConvBenchmark {
  async run(size: number): Promise<number> {
    const math = new NDArrayMathGPU();
    const gpgpu = math.getGPGPUContext();

    const inDepth = this.params.inDepth;
    const inShape: [number, number, number] = [size, size, inDepth];
    const outDepth = this.params.outDepth;
    const filterSize = this.params.filterSize;
    const stride = this.params.stride;

    const x = Array3D.randUniform(inShape, -1, 1);
    const wShape = conv_util.computeWeightsShape4D(
        inDepth, outDepth, filterSize, filterSize);
    const W = Array4D.randUniform(wShape, -1, 1);
    const b = Array1D.randUniform([outDepth], -1, 1);

    let out: NDArray;
    const benchmark = () => {
      out = math.conv2d(x, W, b, stride, 'same');
    };

    const cleanup = () => {
      x.dispose();
      W.dispose();
      b.dispose();
      out.dispose();
    };

    // Warmup.
    await gpgpu.runQuery(benchmark);
    out.dispose();

    let totalTime: number;

    if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE')) {
      totalTime = await gpgpu.runQuery(benchmark);
    } else {
      const start = performance.now();

      benchmark();
      out.dataSync();

      totalTime = performance.now() - start;
    }
    cleanup();
    return totalTime;
  }
}
