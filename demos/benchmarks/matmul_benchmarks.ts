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
// tslint:disable-next-line:max-line-length
import {Array2D, ENV, GPGPUContext, NDArrayMathCPU} from '../deeplearn';

import {BenchmarkTest} from './benchmark';

export class MatmulCPUBenchmark extends BenchmarkTest {
  async run(size: number): Promise<number> {
    if (size > 512) {
      return new Promise<number>((resolve, reject) => {
        resolve(-1);
      });
    }
    const math = new NDArrayMathCPU();
    const a = Array2D.randUniform([size, size], -1, 1);
    const b = Array2D.randUniform([size, size], -1, 1);
    const start = performance.now();
    math.matMul(a, b);
    const end = performance.now();

    return end - start;
  }
}

export class MatmulGPUBenchmark extends BenchmarkTest {
  async run(size: number): Promise<number> {
    const gpgpu = new GPGPUContext();

    const aTexture = gpgpu.createMatrixTexture(size, size);
    const bTexture = gpgpu.createMatrixTexture(size, size);
    const resultTexture = gpgpu.createMatrixTexture(size, size);

    const aArr =
        Array2D.make(
            [size, size], {texture: aTexture, textureShapeRC: [size, size]}) as
        Array2D;
    const bArr =
        Array2D.make(
            [size, size], {texture: bTexture, textureShapeRC: [size, size]}) as
        Array2D;
    const resArr =
        Array2D.make(
            [size, size],
            {texture: resultTexture, textureShapeRC: [size, size]}) as Array2D;
    const program = new MatMulProgram(aArr.shape, bArr.shape);
    const binary =
        gpgpu_math.compileProgram(gpgpu, program, [aArr, bArr], resArr);
    const a = test_util.randomArrayInRange(size * size, -1, 1);
    const b = test_util.randomArrayInRange(size * size, -1, 1);

    gpgpu.uploadMatrixToTexture(aTexture, size, size, a);
    gpgpu.uploadMatrixToTexture(bTexture, size, size, b);

    const benchmark = () => {
      gpgpu_math.runProgram(binary, [aArr, bArr], resArr);
    };

    const cleanup = () => {
      gpgpu.deleteMatrixTexture(aTexture);
      gpgpu.deleteMatrixTexture(bTexture);
      gpgpu.deleteMatrixTexture(resultTexture);
      gpgpu.deleteProgram(binary.webGLProgram);
      gpgpu.dispose();
    };

    // Warmup.
    await gpgpu.runQuery(benchmark);

    let totalTime: number;
    if (ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE')) {
      totalTime = await gpgpu.runQuery(benchmark);
    } else {
      const start = performance.now();

      benchmark();
      resArr.dataSync();

      totalTime = performance.now() - start;
    }

    cleanup();
    return totalTime;
  }
}
