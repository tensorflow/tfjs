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

const CPU_OP_RUNS = 1;

export interface PoolBenchmarkParams {
  depth: number;
  fieldSize: number;
  stride: number;
}

function getPoolingOp(option: string): (
    x: dl.Tensor3D, filterSize: [number, number]|number,
    strides: [number, number]|number) => dl.Tensor3D {
  switch (option) {
    case 'max':
      return (x: dl.Tensor3D, filterSize: [number, number]|number,
              strides: [number, number]|number) => {
        return x.maxPool(filterSize, strides, 'same');
      };
    case 'min':
      return (x: dl.Tensor3D, filterSize: [number, number]|number,
              strides: [number, number]|number) => {
        return x.minPool(filterSize, strides, 'same');
      };
    case 'avg':
      return (x: dl.Tensor3D, filterSize: [number, number]|number,
              strides: [number, number]|number) => {
        return x.avgPool(filterSize, strides, 'same');
      };
    default:
      throw new Error(`Not found such ops: ${option}`);
  }
}

export class PoolCPUBenchmark implements BenchmarkTest {
  run(size: number, option: string,
      params: PoolBenchmarkParams): Promise<number> {
    dl.setBackend('cpu');

    const outputDepth = params.depth;
    const xShape: [number, number, number] = [size, size, outputDepth];
    const fieldSize = params.fieldSize;
    const stride = params.stride;
    const op = getPoolingOp(option);

    const x: dl.Tensor3D = dl.randomUniform(xShape, -1, 1);

    const start = performance.now();
    for (let i = 0; i < CPU_OP_RUNS; i++) {
      op(x, fieldSize, stride);
    }
    const avgTime = (performance.now() - start) / CPU_OP_RUNS;
    return new Promise<number>((resolve, reject) => {
      resolve(avgTime);
    });
  }
}

export class PoolGPUBenchmark implements BenchmarkTest {
  async run(size: number, option: string, params: PoolBenchmarkParams):
      Promise<number> {
    dl.setBackend('webgl');

    const outputDepth = params.depth;
    const xShape: [number, number, number] = [size, size, outputDepth];
    const fieldSize = params.fieldSize;
    const stride = params.stride;
    const x: dl.Tensor3D = dl.randomUniform(xShape, -1, 1);
    const op = getPoolingOp(option);

    const benchmark = () => op(x, fieldSize, stride);
    const time = await benchmark_util.warmupAndBenchmarkGPU(benchmark);

    x.dispose();

    return time;
  }
}
