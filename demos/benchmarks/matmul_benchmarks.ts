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

import * as gpgpu_math from '../../src/math/webgl/gpgpu_math';
import {MatMulProgram} from '../../src/math/webgl/mulmat_gpu';
import * as test_util from '../../src/test_util';
import {Array2D, GPGPUContext, NDArray, NDArrayMathCPU} from '../deeplearn';

import {BenchmarkTest} from './benchmark';

const GPU_OPS_PER_RUN = 40;

export class MatmulCPUBenchmark extends BenchmarkTest {
  run(size: number): number {
    if (size > 512) {
      return -1;
    }
    const math = new NDArrayMathCPU();
    const a = NDArray.randUniform<Array2D>([size, size], -1, 1);
    const b = NDArray.randUniform<Array2D>([size, size], -1, 1);
    const start = performance.now();
    math.matMul(a, b);
    const end = performance.now();
    return end - start;
  }
}

export class MatmulGPUBenchmark extends BenchmarkTest {
  run(size: number): number {
    const gpgpu = new GPGPUContext();
    const aTexture = gpgpu.createMatrixTexture(size, size);
    const bTexture = gpgpu.createMatrixTexture(size, size);
    const resultTexture = gpgpu.createMatrixTexture(size, size);

    const aArr = new Array2D(
        [size, size], {texture: aTexture, textureShapeRC: [size, size]});
    const bArr = new Array2D(
        [size, size], {texture: bTexture, textureShapeRC: [size, size]});
    const resArr = new Array2D(
        [size, size], {texture: resultTexture, textureShapeRC: [size, size]});
    const program = new MatMulProgram(aArr.shape, bArr.shape);
    const binary =
        gpgpu_math.compileProgram(gpgpu, program, [aArr, bArr], resArr);
    const a = test_util.randomArrayInRange(size * size, -1, 1);
    const b = test_util.randomArrayInRange(size * size, -1, 1);
    gpgpu.uploadMatrixToTexture(aTexture, size, size, a);
    gpgpu.uploadMatrixToTexture(bTexture, size, size, b);

    const start = performance.now();
    for (let i = 0; i < GPU_OPS_PER_RUN; i++) {
      gpgpu_math.runProgram(binary, [aArr, bArr], resArr);
    }
    gpgpu.downloadMatrixFromTexture(resultTexture, size, size);
    const avgTime = (performance.now() - start) / GPU_OPS_PER_RUN;

    gpgpu.deleteMatrixTexture(aTexture);
    gpgpu.deleteMatrixTexture(bTexture);
    gpgpu.deleteMatrixTexture(resultTexture);
    gpgpu.deleteProgram(binary.webGLProgram);
    gpgpu.dispose();

    return avgTime;
  }
}
