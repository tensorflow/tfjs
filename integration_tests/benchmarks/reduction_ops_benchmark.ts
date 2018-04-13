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

function getReductionOp(option: string): (x: dl.Tensor) => dl.Scalar {
  switch (option) {
    case 'max':
      return x => x.max();
    case 'min':
      return x => x.min();
    case 'argMax':
      return x => x.argMax();
    case 'argMin':
      return x => x.argMin();
    case 'sum':
      return x => x.sum();
    case 'logSumExp':
      return x => x.logSumExp();
    default:
      throw new Error(`Not found such ops: ${option}`);
  }
}

export class ReductionOpsCPUBenchmark implements BenchmarkTest {
  async run(size: number, option: string): Promise<number> {
    dl.setBackend('cpu');

    const input: dl.Tensor2D = dl.randomUniform([size, size], -1, 1);
    const op = getReductionOp(option);
    const start = performance.now();

    dl.tidy(() => {
      op(input).get();
    });

    const end = performance.now();
    return end - start;
  }
}

export class ReductionOpsGPUBenchmark implements BenchmarkTest {
  async run(size: number, option: string) {
    dl.setBackend('webgl');

    const input: dl.Tensor2D = dl.randomUniform([size, size], -1, 1);
    const op = getReductionOp(option);

    const benchmark = () => op(input);

    const time = await benchmark_util.warmupAndBenchmarkGPU(benchmark);

    input.dispose();

    return time;
  }
}
