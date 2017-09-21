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

import {NDArrayMath} from '../../src/math/math';
import {NDArrayMathCPU} from '../../src/math/math_cpu';
import {NDArrayMathGPU} from '../../src/math/math_gpu';
import {Array2D, NDArray} from '../../src/math/ndarray';
import {Scalar} from '../../src/math/ndarray';

import {BenchmarkTest} from './benchmark';

const OPS_PER_RUN = 10;

export abstract class ReductionOpsBenchmark extends BenchmarkTest {

  protected getReductionOp(option: string,
                           math: NDArrayMath): (input: NDArray) => Scalar {
    switch (option) {
    case "max":
      return (input: NDArray) => math.max(input);
    case "min":
      return (input: NDArray) => math.min(input);
    case "sum":
      return (input: NDArray) => math.sum(input);
    default:
      throw new Error(`Not found such ops: ${option}`);
    }
  }
}

export class ReductionOpsCPUBenchmark extends ReductionOpsBenchmark {
  run(size: number, option: string) {
    const math = new NDArrayMathCPU();
    const input = NDArray.randUniform<Array2D>([ size, size ], -1, 1);
    const op = this.getReductionOp(option, math);
    const start = performance.now();

    for (let i = 0; i < OPS_PER_RUN; i++) {
      math.scope(() => { op(input).get(); });
    }
    const end = performance.now();
    return (end - start) / OPS_PER_RUN;
  }
}

export class ReductionOpsGPUBenchmark extends ReductionOpsBenchmark {
  run(size: number, option: string) {
    const math = new NDArrayMathGPU();
    const input = NDArray.randUniform<Array2D>([ size, size ], -1, 1);
    const op = this.getReductionOp(option, math);
    const start = performance.now();

    for (let i = 0; i < OPS_PER_RUN; i++) {
      math.scope(() => { op(input).get(); });
    }
    const end = performance.now();
    return (end - start) / OPS_PER_RUN;
  }
}