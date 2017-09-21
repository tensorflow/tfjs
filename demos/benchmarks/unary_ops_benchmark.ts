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
import {BenchmarkTest} from './benchmark';

const OPS_PER_RUN = 10;

export abstract class UnaryOpsBenchmark extends BenchmarkTest {

  protected getUnaryOp(option: string, math: NDArrayMath) {
    switch (option) {
    case "log":
      return (input: NDArray) => math.log(input);
    case "exp":
      return (input: NDArray) => math.exp(input);
    case "neg":
      return (input: NDArray) => math.neg(input);
    case "sqrt":
      return (input: NDArray) => math.sqrt(input);
    case "abs":
      return (input: NDArray) => math.abs(input);
    case "relu":
      return (input: NDArray) => math.relu(input);
    case "sigmoid":
      return (input: NDArray) => math.sigmoid(input);
    case "sin":
      return (input: NDArray) => math.sin(input);
    case "cos":
      return (input: NDArray) => math.cos(input);
    case "tan":
      return (input: NDArray) => math.tan(input);
    case "asin":
      return (input: NDArray) => math.asin(input);
    case "acos":
      return (input: NDArray) => math.acos(input);
    case "atan":
      return (input: NDArray) => math.atan(input);
    case "sinh":
      return (input: NDArray) => math.sinh(input);
    case "cosh":
      return (input: NDArray) => math.cosh(input);
    case "tanh":
      return (input: NDArray) => math.tanh(input);
    case "logSumExp":
      return (input: NDArray) => math.logSumExp(input);
    default:
      throw new Error(`Not found such ops: ${option}`);
    }
  }
}

export class UnaryOpsCPUBenchmark extends UnaryOpsBenchmark {
  run(size: number, option: string) {
    const math = new NDArrayMathCPU();
    const input = NDArray.randUniform<Array2D>([ size, size ], -1, 1);
    const op = this.getUnaryOp(option, math);
    const start = performance.now();

    for (let i = 0; i < OPS_PER_RUN; i++) {
      math.scope(() => { op(input).get(); });
    }
    const end = performance.now();
    return (end - start) / OPS_PER_RUN;
  }
}

export class UnaryOpsGPUBenchmark extends UnaryOpsBenchmark {
  run(size: number, option: string) {
    const math = new NDArrayMathGPU();
    const input = NDArray.randUniform<Array2D>([ size, size ], -1, 1);
    const op = this.getUnaryOp(option, math);
    const start = performance.now();

    for (let i = 0; i < OPS_PER_RUN; i++) {
      math.scope(() => { op(input).get(); });
    }
    const end = performance.now();
    return (end - start) / OPS_PER_RUN;
  }
}