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
import {Array2D, ENV, NDArray, NDArrayMath, Scalar} from 'deeplearn';

import {BenchmarkTest} from './benchmark';
import * as benchmark_util from './benchmark_util';

function getReductionOp(option: string, math: NDArrayMath): (input: NDArray) =>
    Scalar {
  switch (option) {
    case 'max':
      return input => math.max(input);
    case 'min':
      return input => math.min(input);
    case 'argMax':
      return input => math.argMax(input) as Scalar;
    case 'argMin':
      return input => math.argMin(input) as Scalar;
    case 'sum':
      return input => math.sum(input) as Scalar;
    case 'logSumExp':
      return input => math.logSumExp(input) as Scalar;
    default:
      throw new Error(`Not found such ops: ${option}`);
  }
}

export class ReductionOpsCPUBenchmark implements BenchmarkTest {
  async run(size: number, option: string): Promise<number> {
    const safeMode = false;
    const math = new NDArrayMath('cpu', safeMode);
    ENV.setMath(math);
    const input = Array2D.randUniform([size, size], -1, 1);
    const op = getReductionOp(option, math);
    const start = performance.now();

    math.scope(() => {
      op(input).get();
    });

    const end = performance.now();
    return end - start;
  }
}

export class ReductionOpsGPUBenchmark implements BenchmarkTest {
  async run(size: number, option: string) {
    const safeMode = false;
    const math = new NDArrayMath('webgl', safeMode);
    ENV.setMath(math);
    const input = Array2D.randUniform([size, size], -1, 1);
    const op = getReductionOp(option, math);

    const benchmark = () => op(input);

    const time = await benchmark_util.warmupAndBenchmarkGPU(math, benchmark);

    input.dispose();

    return time;
  }
}
