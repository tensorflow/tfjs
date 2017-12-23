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
import {Array2D, ENV, NDArrayMath} from 'deeplearn';

import {BenchmarkTest, LAST_RUN_CPU_CUTOFF_MS} from './benchmark';
import * as benchmark_util from './benchmark_util';

export class MatmulCPUBenchmark implements BenchmarkTest {
  lastRunTimeMs: number;
  async run(size: number): Promise<number> {
    if (this.lastRunTimeMs > LAST_RUN_CPU_CUTOFF_MS) {
      return new Promise<number>((resolve, reject) => {
        resolve(-1);
      });
    }
    const safeMode = false;
    const math = new NDArrayMath('cpu', safeMode);
    ENV.setMath(math);
    const a = Array2D.randUniform([size, size], -1, 1);
    const b = Array2D.randUniform([size, size], -1, 1);
    const start = performance.now();
    math.matMul(a, b);
    const end = performance.now();

    this.lastRunTimeMs = end - start;
    return this.lastRunTimeMs;
  }
}

export class MatmulGPUBenchmark implements BenchmarkTest {
  async run(size: number): Promise<number> {
    const safeMode = false;
    const math = new NDArrayMath('webgl', safeMode);
    ENV.setMath(math);
    const a = Array2D.randNormal([size, size]);
    const b = Array2D.randNormal([size, size]);

    const benchmark = () => math.matMul(a, b);

    const time = await benchmark_util.warmupAndBenchmarkGPU(math, benchmark);

    a.dispose();
    b.dispose();
    math.dispose();

    return time;
  }
}
