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
import {Array3D, Array4D, conv_util, ENV, NDArray, NDArrayMathGPU} from 'deeplearn';

import {BenchmarkTest} from './benchmark';

export interface ConvTransposedBenchmarkParams {
  inDepth: number;
  outDepth: number;
  filterSize: number;
  stride: number;
}

export abstract class ConvTransposedBenchmark extends BenchmarkTest {
  constructor(protected params: ConvTransposedBenchmarkParams) {
    super(params);
  }
}

export class ConvTransposedGPUBenchmark extends ConvTransposedBenchmark {
  async run(size: number): Promise<number> {
    const math = new NDArrayMathGPU();
    const gpgpu = math.getGPGPUContext();

    const origInputDepth = 1;
    const origOutputDepth = 1;
    const xShape: [number, number, number] = [size, size, origOutputDepth];
    const fieldSize = 11;
    const origStride = 1;
    const origPad = 1;

    const x = Array3D.randUniform(xShape, -1, 1);
    const wShape = conv_util.computeWeightsShape4D(
        origInputDepth, origOutputDepth, fieldSize, fieldSize);
    const W = Array4D.randUniform(wShape, -1, 1);

    let out: NDArray;
    const benchmark = () => {
      out = math.conv2dTranspose(
          x, W, [size, size, origInputDepth], origStride, origPad);
    };

    const cleanup = () => {
      out.dispose();
      x.dispose();
      W.dispose();
      gpgpu.dispose();
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
