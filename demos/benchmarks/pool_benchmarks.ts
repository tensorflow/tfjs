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

import {initializeGPU} from '../../src/math/ndarray';
import * as gpgpu_math from '../../src/math/webgl/gpgpu_math';
import {Pool2DProgram} from '../../src/math/webgl/pool_gpu';
import {TextureManager} from '../../src/math/webgl/texture_manager';
// tslint:disable-next-line:max-line-length
import {Array3D, conv_util, ENV, GPGPUContext, NDArray, NDArrayMathCPU} from '../deeplearn';

import {BenchmarkTest} from './benchmark';

const CPU_OP_RUNS = 1;

export interface PoolBenchmarkParams {
  depth: number;
  fieldSize: number;
  stride: number;
  type: 'max'|'avg';
}

export abstract class PoolBenchmark extends BenchmarkTest {
  constructor(protected params: PoolBenchmarkParams) {
    super(params);
  }
}

export class PoolCPUBenchmark extends PoolBenchmark {
  run(size: number): Promise<number> {
    const math = new NDArrayMathCPU();
    const outputDepth = this.params.depth;
    const xShape: [number, number, number] = [size, size, outputDepth];
    const fieldSize = this.params.fieldSize;
    const stride = this.params.stride;
    const zeroPad = conv_util.computeDefaultPad(xShape, fieldSize, stride);

    const x = Array3D.randUniform(xShape, -1, 1);

    const start = performance.now();
    for (let i = 0; i < CPU_OP_RUNS; i++) {
      math.maxPool(x as Array3D, fieldSize, stride, zeroPad);
    }
    const avgTime = (performance.now() - start) / CPU_OP_RUNS;

    return new Promise<number>((resolve, reject) => {
      resolve(avgTime);
    });
  }
}

export class PoolGPUBenchmark extends PoolBenchmark {
  run(size: number): Promise<number> {
    return new Promise<number>((resolve, reject) => {

      const gpgpu = new GPGPUContext();
      const texManager = new TextureManager(gpgpu);
      initializeGPU(gpgpu, texManager);

      const outputDepth = this.params.depth;
      const xShape: [number, number, number] = [size, size, outputDepth];
      const fieldSize = this.params.fieldSize;
      const stride = this.params.stride;
      const convInfo = conv_util.computeConvInfo(
          xShape, fieldSize, fieldSize, outputDepth, stride, stride, 'same');
      const program = new Pool2DProgram(convInfo, this.params.type, false);
      const res = NDArray.zeros(program.outputShape);
      const x = Array3D.randUniform(xShape, -1, 1);
      const binary = gpgpu_math.compileProgram(gpgpu, program, [x], res);

      const benchmark = () => {
        gpgpu_math.runProgram(binary, [x], res);
      };

      const immediateCleanup = () => {
        x.dispose();
        res.dispose();
        texManager.dispose();
        gpgpu.deleteProgram(binary.webGLProgram);
      };

      const delayedCleanup = () => {
        gpgpu.dispose();
      };

      if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER')) {
        gpgpu.runBenchmark(benchmark).then((timeElapsed: number) => {
          delayedCleanup();
          resolve(timeElapsed);
        });
        immediateCleanup();
      } else {
        const start = performance.now();

        benchmark();
        res.getValues();

        const totalTime = performance.now() - start;

        immediateCleanup();
        delayedCleanup();

        resolve(totalTime);
      }
    });
  }
}
