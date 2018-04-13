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

function getUnaryOp(option: string) {
  switch (option) {
    case 'log':
      return (x: dl.Tensor) => x.log();
    case 'exp':
      return (x: dl.Tensor) => x.exp();
    case 'neg':
      return (x: dl.Tensor) => x.neg();
    case 'ceil':
      return (x: dl.Tensor) => x.ceil();
    case 'floor':
      return (x: dl.Tensor) => x.floor();
    case 'log1p':
      return (x: dl.Tensor) => x.log1p();
    case 'sqrt':
      return (x: dl.Tensor) => x.sqrt();
    case 'square':
      return (x: dl.Tensor) => x.square();
    case 'abs':
      return (x: dl.Tensor) => x.abs();
    case 'relu':
      return (x: dl.Tensor) => x.relu();
    case 'elu':
      return (x: dl.Tensor) => x.elu();
    case 'selu':
      return (x: dl.Tensor) => x.selu();
    case 'leakyRelu':
      return (x: dl.Tensor) => x.leakyRelu();
    case 'prelu':
      // TODO: Configurable from UI
      const alpha = dl.scalar(0.1);
      return (x: dl.Tensor) => x.prelu(alpha);
    case 'sigmoid':
      return (x: dl.Tensor) => x.sigmoid();
    case 'sin':
      return (x: dl.Tensor) => x.sin();
    case 'cos':
      return (x: dl.Tensor) => x.cos();
    case 'tan':
      return (x: dl.Tensor) => x.tan();
    case 'asin':
      return (x: dl.Tensor) => x.asin();
    case 'acos':
      return (x: dl.Tensor) => x.acos();
    case 'atan':
      return (x: dl.Tensor) => x.atan();
    case 'sinh':
      return (x: dl.Tensor) => x.sinh();
    case 'cosh':
      return (x: dl.Tensor) => x.cosh();
    case 'tanh':
      return (x: dl.Tensor) => x.tanh();
    case 'step':
      return (x: dl.Tensor) => x.step();
    default:
      throw new Error(`Not found such ops: ${option}`);
  }
}

export class UnaryOpsCPUBenchmark implements BenchmarkTest {
  async run(size: number, option: string): Promise<number> {
    dl.setBackend('cpu');

    const input: dl.Tensor2D = dl.randomUniform([size, size], -1, 1);
    const op = getUnaryOp(option);
    const start = performance.now();

    dl.tidy(() => {
      op(input).get();
    });

    const end = performance.now();
    return end - start;
  }
}

export class UnaryOpsGPUBenchmark implements BenchmarkTest {
  async run(size: number, option: string) {
    dl.setBackend('webgl');

    const input: dl.Tensor2D = dl.randomUniform([size, size], -1, 1);
    const op = getUnaryOp(option);

    const benchmark = () => op(input);

    const time = await benchmark_util.warmupAndBenchmarkGPU(benchmark);

    input.dispose();
    return time;
  }
}
