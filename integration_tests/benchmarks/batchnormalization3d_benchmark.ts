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

import {BenchmarkTest, LAST_RUN_CPU_CUTOFF_MS} from './benchmark';
import * as benchmark_util from './benchmark_util';

export class BatchNormalization3DCPUBenchmark implements BenchmarkTest {
  lastRunTimeMs: number;

  async run(size: number): Promise<number> {
    if (this.lastRunTimeMs > LAST_RUN_CPU_CUTOFF_MS) {
      return new Promise<number>((resolve, reject) => {
        resolve(-1);
      });
    }
    dl.setBackend('cpu');
    const x: dl.Tensor3D = dl.randomUniform([size, size, 8], -1, 1);
    const mean = dl.tensor1d([0]);
    const variance = dl.tensor1d([1]);
    const varianceEpsilon = .001;
    const start = performance.now();

    x.batchNormalization(mean, variance, varianceEpsilon);

    const end = performance.now();

    this.lastRunTimeMs = end - start;
    return this.lastRunTimeMs;
  }
}

export class BatchNormalization3DGPUBenchmark implements BenchmarkTest {
  async run(size: number) {
    dl.setBackend('webgl');

    const x: dl.Tensor3D = dl.randomUniform([size, size, 8], -1, 1);
    const mean = dl.tensor1d([0]);
    const variance = dl.tensor1d([1]);
    const varianceEpsilon = .001;

    const benchmark = () =>
        x.batchNormalization(mean, variance, varianceEpsilon);

    const time = await benchmark_util.warmupAndBenchmarkGPU(benchmark);

    x.dispose();
    mean.dispose();
    variance.dispose();

    return time;
  }
}
