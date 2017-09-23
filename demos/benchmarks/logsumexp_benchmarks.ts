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
import {LogSumExpProgram} from '../../src/math/webgl/logsumexp_gpu';
import {TextureManager} from '../../src/math/webgl/texture_manager';
// tslint:disable-next-line:max-line-length
import {Array2D, ENV, GPGPUContext, NDArray, NDArrayMathCPU, Scalar} from '../deeplearn';

import {BenchmarkTest} from './benchmark';

export class LogSumExpCPUBenchmark extends BenchmarkTest {
  run(size: number): Promise<number> {
    const math = new NDArrayMathCPU();
    const a = NDArray.randUniform<Array2D>([size, size], -1, 1);
    const start = performance.now();
    math.logSumExp(a);

    const end = performance.now();

    return new Promise<number>((resolve, reject) => {
      resolve(end - start);
    });
  }
}

export class LogSumExpGPUBenchmark extends BenchmarkTest {
  run(size: number): Promise<number> {
    return new Promise<number>((resolve, reject) => {
      const gpgpu = new GPGPUContext();
      const texManager = new TextureManager(gpgpu);
      initializeGPU(gpgpu, texManager);
      const out = new Scalar({texture: texManager.acquireTexture([1, 1])});
      const a = Array2D.randUniform([size, size], -1, 1);
      const program = new LogSumExpProgram(a.size);
      const binary = gpgpu_math.compileProgram(gpgpu, program, [a], out);

      const benchmark = () => {
        gpgpu_math.runProgram(binary, [a], out);
      };

      const immediateCleanup = () => {
        a.dispose();
        out.dispose();
        texManager.dispose();
        gpgpu.deleteProgram(binary.webGLProgram);
        gpgpu.deleteProgram(binary.webGLProgram);
      };

      const delayedCleanup = () => {
        gpgpu.dispose();
      };

      if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE')) {
        gpgpu.runQuery(benchmark).then((timeElapsed: number) => {
          delayedCleanup();
          resolve(timeElapsed);
        });
        immediateCleanup();
      } else {
        const start = performance.now();

        benchmark();
        out.getValues();

        const totalTime = performance.now() - start;

        immediateCleanup();
        delayedCleanup();

        resolve(totalTime);
      }
    });
  }
}
