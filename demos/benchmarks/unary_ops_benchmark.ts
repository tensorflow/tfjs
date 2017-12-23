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
import {Array2D, ENV, NDArray, NDArrayMath} from 'deeplearn';

import {BenchmarkTest} from './benchmark';
import * as benchmark_util from './benchmark_util';

function getUnaryOp(option: string, math: NDArrayMath) {
  switch (option) {
    case 'log':
      return (input: NDArray) => math.log(input);
    case 'exp':
      return (input: NDArray) => math.exp(input);
    case 'neg':
      return (input: NDArray) => math.neg(input);
    case 'sqrt':
      return (input: NDArray) => math.sqrt(input);
    case 'abs':
      return (input: NDArray) => math.abs(input);
    case 'relu':
      return (input: NDArray) => math.relu(input);
    case 'sigmoid':
      return (input: NDArray) => math.sigmoid(input);
    case 'sin':
      return (input: NDArray) => math.sin(input);
    case 'cos':
      return (input: NDArray) => math.cos(input);
    case 'tan':
      return (input: NDArray) => math.tan(input);
    case 'asin':
      return (input: NDArray) => math.asin(input);
    case 'acos':
      return (input: NDArray) => math.acos(input);
    case 'atan':
      return (input: NDArray) => math.atan(input);
    case 'sinh':
      return (input: NDArray) => math.sinh(input);
    case 'cosh':
      return (input: NDArray) => math.cosh(input);
    case 'tanh':
      return (input: NDArray) => math.tanh(input);
    default:
      throw new Error(`Not found such ops: ${option}`);
  }
}

export class UnaryOpsCPUBenchmark implements BenchmarkTest {
  async run(size: number, option: string): Promise<number> {
    const safeMode = false;
    const math = new NDArrayMath('cpu', safeMode);
    ENV.setMath(math);
    const input = Array2D.randUniform([size, size], -1, 1);
    const op = getUnaryOp(option, math);
    const start = performance.now();

    math.scope(() => {
      op(input).get();
    });

    const end = performance.now();
    return end - start;
  }
}

export class UnaryOpsGPUBenchmark implements BenchmarkTest {
  async run(size: number, option: string) {
    const safeMode = false;
    const math = new NDArrayMath('webgl', safeMode);
    ENV.setMath(math);
    const input = Array2D.randUniform([size, size], -1, 1);
    const op = getUnaryOp(option, math);

    const benchmark = () => op(input);

    const time = await benchmark_util.warmupAndBenchmarkGPU(math, benchmark);

    input.dispose();
    math.dispose();

    return time;
  }
}
