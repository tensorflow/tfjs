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
import {Array2D, ENV, NDArray, NDArrayMath, NDArrayMathCPU, NDArrayMathGPU, Scalar} from 'deeplearn';

import {BenchmarkTest} from './benchmark';

export abstract class ReductionOpsBenchmark extends BenchmarkTest {
  protected getReductionOp(option: string, math: NDArrayMath):
      (input: NDArray) => Scalar {
    switch (option) {
      case 'max':
        return input => math.max(input);
      case 'min':
        return input => math.min(input);
      case 'argMax':
        return input => math.argMax(input);
      case 'argMin':
        return input => math.argMin(input);
      case 'sum':
        return input => math.sum(input);
      case 'logSumExp':
        return input => math.logSumExp(input);
      default:
        throw new Error(`Not found such ops: ${option}`);
    }
  }
}

export class ReductionOpsCPUBenchmark extends ReductionOpsBenchmark {
  async run(size: number, option: string): Promise<number> {
    const math = new NDArrayMathCPU();
    const input = Array2D.randUniform([size, size], -1, 1);
    const op = this.getReductionOp(option, math);
    const start = performance.now();

    math.scope(() => {
      op(input).get();
    });

    const end = performance.now();
    return end - start;
  }
}

export class ReductionOpsGPUBenchmark extends ReductionOpsBenchmark {
  async run(size: number, option: string) {
    const math = new NDArrayMathGPU();
    const input = Array2D.randUniform([size, size], -1, 1);
    const op = this.getReductionOp(option, math);

    let output: NDArray;
    const benchmark = () => {
      math.scope(() => {
        output = op(input);
      });
    };

    const cleanup = () => {
      input.dispose();
      math.dispose();
    };

    // Warmup.
    await math.getGPGPUContext().runQuery(benchmark);

    let totalTime: number;
    if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE')) {
      totalTime = await math.getGPGPUContext().runQuery(benchmark);
    } else {
      const start = performance.now();

      benchmark();
      output.dataSync();

      totalTime = performance.now() - start;

      cleanup();
    }

    cleanup();

    return totalTime;
  }
}
