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
import {Array2D, ENV, NDArray, NDArrayMath, NDArrayMathCPU, NDArrayMathGPU, Scalar} from '../deeplearn';

import {BenchmarkTest} from './benchmark';

export abstract class ReductionOpsBenchmark extends BenchmarkTest {
  protected getReductionOp(option: string, math: NDArrayMath):
      (input: NDArray) => Scalar {
    switch (option) {
      case 'max':
        return (input: NDArray) => math.max(input);
      case 'min':
        return (input: NDArray) => math.min(input);
      case 'sum':
        return (input: NDArray) => math.sum(input);
      default:
        throw new Error(`Not found such ops: ${option}`);
    }
  }
}

export class ReductionOpsCPUBenchmark extends ReductionOpsBenchmark {
  run(size: number, option: string): Promise<number> {
    return new Promise<number>((resolve, reject) => {
      const math = new NDArrayMathCPU();
      const input = Array2D.randUniform([size, size], -1, 1);
      const op = this.getReductionOp(option, math);
      const start = performance.now();

      math.scope(() => {
        op(input).get();
      });

      const end = performance.now();
      resolve(end - start);
    });
  }
}

export class ReductionOpsGPUBenchmark extends ReductionOpsBenchmark {
  run(size: number, option: string) {
    return new Promise<number>((resolve, reject) => {
      const math = new NDArrayMathGPU();
      const input = Array2D.randUniform([size, size], -1, 1);
      const op = this.getReductionOp(option, math);

      let output: NDArray;
      const benchmark = () => {
        math.scope(() => {
          output = op(input);
        });
      };

      const immediateCleanup = () => {
        input.dispose();
      };

      const delayedCleanup = () => {
        math.dispose();
      };

      if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE')) {
        math.getGPGPUContext().runQuery(benchmark).then(
            (timeElapsed: number) => {
              delayedCleanup();
              resolve(timeElapsed);
            });
        immediateCleanup();
      } else {
        const start = performance.now();

        benchmark();
        output.get();

        const totalTime = performance.now() - start;

        immediateCleanup();
        delayedCleanup();

        resolve(totalTime);
      }
    });
  }
}
