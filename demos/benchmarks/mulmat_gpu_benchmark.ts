/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import {MatrixOrientation} from '../../src/math/math';
import {Array2D} from '../../src/math/ndarray';
import {GPGPUContext} from '../../src/math/webgl/gpgpu_context';
import {MatMulProgram} from '../../src/math/webgl/mulmat_gpu';
import * as gpgpu_math from '../../src/math/webgl/gpgpu_math';
import * as mulmat_packed_gpu from '../../src/math/webgl/mulmat_packed_gpu';
import * as test_util from '../../src/test_util';

import {BenchmarkTest} from './benchmark';

const OP_RUNS = 40;

export const BENCHMARK_TEST: BenchmarkTest = (size: number) => {
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
  for (let i = 0; i < OP_RUNS; i++) {
    gpgpu_math.runProgram(binary, [aArr, bArr], resArr);
  }
  gpgpu.downloadMatrixFromTexture(resultTexture, size, size);
  const avgTime = (performance.now() - start) / OP_RUNS;

  gpgpu.deleteMatrixTexture(aTexture);
  gpgpu.deleteMatrixTexture(bTexture);
  gpgpu.deleteMatrixTexture(resultTexture);
  gpgpu.deleteProgram(binary.webGLProgram);
  gpgpu.dispose();

  return avgTime;
};

export const BENCHMARK_TEST_PACKED: BenchmarkTest = (size: number) => {
  const gpgpu = new GPGPUContext();
  const program: WebGLProgram =
      gpgpu.createProgram(mulmat_packed_gpu.getFragmentShaderSource(
          size, MatrixOrientation.REGULAR, MatrixOrientation.REGULAR));

  const aTexture = gpgpu.createPackedMatrixTexture(size, size);
  const bTexture = gpgpu.createPackedMatrixTexture(size, size);
  const resultTexture = gpgpu.createPackedMatrixTexture(size, size);

  const a = test_util.randomArrayInRange(size * size, -1, 1);
  const b = test_util.randomArrayInRange(size * size, -1, 1);
  gpgpu.uploadMatrixToPackedTexture(aTexture, size, size, a);
  gpgpu.uploadMatrixToPackedTexture(bTexture, size, size, b);

  const start = performance.now();
  for (let i = 0; i < OP_RUNS; i++) {
    mulmat_packed_gpu.multiplyMatrixPacked(
        gpgpu, program, aTexture, bTexture, resultTexture, [size, size]);
  }

  gpgpu.downloadMatrixFromPackedTexture(resultTexture, size, size);
  const avgTime = (performance.now() - start) / OP_RUNS;

  gpgpu.deleteMatrixTexture(aTexture);
  gpgpu.deleteMatrixTexture(bTexture);
  gpgpu.deleteMatrixTexture(resultTexture);
  gpgpu.deleteProgram(program);
  gpgpu.dispose();

  return avgTime;
};
