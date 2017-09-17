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
import {Array2D, GPGPUContext, NDArray, NDArrayMathCPU, Scalar} from '../deeplearn';

import {BenchmarkTest} from './benchmark';

const CPU_OPS_PER_RUN = 10;
const GPU_OPS_PER_RUN = 10;

export class LogSumExpCPUBenchmark extends BenchmarkTest {
  run(size: number): number {
    const math = new NDArrayMathCPU();
    const a = NDArray.randUniform<Array2D>([size, size], -1, 1);
    const start = performance.now();
    for (let i = 0; i < CPU_OPS_PER_RUN; i++) {
      math.logSumExp(a);
    }
    const end = performance.now();
    return (end - start) / CPU_OPS_PER_RUN;
  }
}

export class LogSumExpGPUBenchmark extends BenchmarkTest {
  run(size: number): number {
    const gpgpu = new GPGPUContext();
    const texManager = new TextureManager(gpgpu);
    initializeGPU(gpgpu, texManager);
    const out = new Scalar({texture: texManager.acquireTexture([1, 1])});
    const a = Array2D.randUniform([size, size], -1, 1);
    const program = new LogSumExpProgram(a.size);
    const binary = gpgpu_math.compileProgram(gpgpu, program, [a], out);

    const start = performance.now();
    for (let i = 0; i < GPU_OPS_PER_RUN; i++) {
      gpgpu_math.runProgram(binary, [a], out);
    }
    out.getValues();
    const avgTime = (performance.now() - start) / GPU_OPS_PER_RUN;
    a.dispose();
    out.dispose();
    texManager.dispose();
    gpgpu.deleteProgram(binary.webGLProgram);
    gpgpu.dispose();

    return avgTime;
  }
}
