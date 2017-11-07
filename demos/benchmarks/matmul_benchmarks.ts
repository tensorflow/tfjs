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
import {Array2D, ENV, NDArray, NDArrayMathCPU, NDArrayMathGPU} from 'deeplearn';

import {BenchmarkTest} from './benchmark';

export class MatmulCPUBenchmark extends BenchmarkTest {
  async run(size: number): Promise<number> {
    if (size > 512) {
      return new Promise<number>((resolve, reject) => {
        resolve(-1);
      });
    }
    const math = new NDArrayMathCPU();
    const a = Array2D.randUniform([size, size], -1, 1);
    const b = Array2D.randUniform([size, size], -1, 1);
    const start = performance.now();
    math.matMul(a, b);
    const end = performance.now();

    return end - start;
  }
}

export class MatmulGPUBenchmark extends BenchmarkTest {
  async run(size: number): Promise<number> {
    const math = new NDArrayMathGPU();
    const gpgpu = math.getGPGPUContext();

    const a = Array2D.randNormal([size, size]);
    const b = Array2D.randNormal([size, size]);

    let out: NDArray;
    const benchmark = () => {
      out = math.matMul(a, b);
    };

    const cleanup = () => {
      a.dispose();
      b.dispose();
      out.dispose();
      math.dispose();
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
