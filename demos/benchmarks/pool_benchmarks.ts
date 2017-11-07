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
import {Array3D, conv_util, ENV, NDArray, NDArrayMath, NDArrayMathCPU, NDArrayMathGPU} from 'deeplearn';
import {BenchmarkTest} from './benchmark';

const CPU_OP_RUNS = 1;

export interface PoolBenchmarkParams {
  depth: number;
  fieldSize: number;
  stride: number;
  type: 'max'|'min'|'avg';
}

export abstract class PoolBenchmark extends BenchmarkTest {
  constructor(protected params: PoolBenchmarkParams) {
    super(params);
  }

  protected getPoolingOp(option: string, math: NDArrayMath):
      (x: Array3D, filterSize: [number, number]|number,
       strides: [number, number]|number,
       pad: 'valid'|'same'|number) => Array3D {
    switch (option) {
      case 'max':
        return (x: Array3D, filterSize: [number, number]|number,
                strides: [number, number]|number,
                pad: 'valid'|'same'|number) => {
          return math.maxPool(x, filterSize, strides, pad);
        };
      case 'min':
        return (x: Array3D, filterSize: [number, number]|number,
                strides: [number, number]|number,
                pad: 'valid'|'same'|number) => {
          return math.minPool(x, filterSize, strides, pad);
        };
      case 'avg':
        return (x: Array3D, filterSize: [number, number]|number,
                strides: [number, number]|number,
                pad: 'valid'|'same'|number) => {
          return math.avgPool(x, filterSize, strides, pad);
        };
      default:
        throw new Error(`Not found such ops: ${option}`);
    }
  }
}

export class PoolCPUBenchmark extends PoolBenchmark {
  run(size: number, option: string): Promise<number> {
    const math = new NDArrayMathCPU();
    const outputDepth = this.params.depth;
    const xShape: [number, number, number] = [size, size, outputDepth];
    const fieldSize = this.params.fieldSize;
    const stride = this.params.stride;
    const zeroPad = conv_util.computeDefaultPad(xShape, fieldSize, stride);
    const op = this.getPoolingOp(option, math);

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

export class PoolGPUBenchmark extends PoolBenchmark {
  async run(size: number, option: string): Promise<number> {
    const math = new NDArrayMathGPU();
    const gpgpu = math.getGPGPUContext();

    const outputDepth = this.params.depth;
    const xShape: [number, number, number] = [size, size, outputDepth];
    const fieldSize = this.params.fieldSize;
    const stride = this.params.stride;
    const x = Array3D.randUniform(xShape, -1, 1);
    const op = this.getPoolingOp(option, math);

    let out: NDArray;
    const benchmark = () => {
      out = op(x, fieldSize, stride, 'same');
    };

    const cleanup = () => {
      x.dispose();
      out.dispose();
      gpgpu.dispose();
    };

    // Warmup.
    await gpgpu.runQuery(benchmark);
    out.dispose();

    let totalTime: number;
    if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE')) {
      totalTime = await gpgpu.runQuery(benchmark);
    } else {
      const start = performance.now();

      benchmark();
      out.dataSync();

      totalTime = performance.now() - start;
    }

    cleanup();

    return totalTime;
  }
}
