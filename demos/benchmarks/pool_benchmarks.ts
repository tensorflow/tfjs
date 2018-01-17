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
import {Array3D, conv_util, ENV, NDArrayMath} from 'deeplearn';
import {BenchmarkTest} from './benchmark';
import * as benchmark_util from './benchmark_util';

const CPU_OP_RUNS = 1;

export interface PoolBenchmarkParams {
  depth: number;
  fieldSize: number;
  stride: number;
}

function getPoolingOp(option: string, math: NDArrayMath): (
    x: Array3D, filterSize: [number, number]|number,
    strides: [number, number]|number, pad: 'valid'|'same'|number) => Array3D {
  switch (option) {
    case 'max':
      return (x: Array3D, filterSize: [number, number]|number,
              strides: [number, number]|number, pad: 'valid'|'same'|number) => {
        return math.maxPool(x, filterSize, strides, pad);
      };
    case 'min':
      return (x: Array3D, filterSize: [number, number]|number,
              strides: [number, number]|number, pad: 'valid'|'same'|number) => {
        return math.minPool(x, filterSize, strides, pad);
      };
    case 'avg':
      return (x: Array3D, filterSize: [number, number]|number,
              strides: [number, number]|number, pad: 'valid'|'same'|number) => {
        return math.avgPool(x.asType('float32'), filterSize, strides, pad);
      };
    default:
      throw new Error(`Not found such ops: ${option}`);
  }
}

export class PoolCPUBenchmark implements BenchmarkTest {
  run(size: number, option: string,
      params: PoolBenchmarkParams): Promise<number> {
    const safeMode = false;
    const math = new NDArrayMath('cpu', safeMode);
    ENV.setMath(math);
    const outputDepth = params.depth;
    const xShape: [number, number, number] = [size, size, outputDepth];
    const fieldSize = params.fieldSize;
    const stride = params.stride;
    const zeroPad = conv_util.computeDefaultPad(xShape, fieldSize, stride);
    const op = getPoolingOp(option, math);

    const x = Array3D.randUniform(xShape, -1, 1);

    const start = performance.now();
    for (let i = 0; i < CPU_OP_RUNS; i++) {
      op(x as Array3D, fieldSize, stride, zeroPad);
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
    const safeMode = false;
    const math = new NDArrayMath('webgl', safeMode);
    ENV.setMath(math);
    const outputDepth = params.depth;
    const xShape: [number, number, number] = [size, size, outputDepth];
    const fieldSize = params.fieldSize;
    const stride = params.stride;
    const x = Array3D.randUniform(xShape, -1, 1);
    const op = getPoolingOp(option, math);

    const benchmark = () => op(x, fieldSize, stride, 'same');
    const time = await benchmark_util.warmupAndBenchmarkGPU(math, benchmark);
    x.dispose();

    return time;
  }
}
