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
import {Array3D, conv_util, GPGPUContext, NDArray} from '../deeplearn';

import {BenchmarkTest} from './benchmark';

const OP_RUNS = 40;

export const MAX_POOL_BENCHMARK_TEST: BenchmarkTest = (size: number) => {
  const positions = false;
  return testMaxPool(size, positions);
};

export const MAX_POOL_POSNS_BENCHMARK_TEST: BenchmarkTest = (size: number) => {
  const positions = true;
  return testMaxPool(size, positions);
};

function testMaxPool(size: number, positions: boolean): number {
  const gpgpu = new GPGPUContext();
  const texManager = new TextureManager(gpgpu);
  initializeGPU(gpgpu, texManager);

  const outputDepth = 1;
  const xShape: [number, number, number] = [size, size, outputDepth];
  const fieldSize = 11;
  const stride = 1;
  const convInfo = conv_util.computeConvInfo(
      xShape, fieldSize, fieldSize, outputDepth, stride, stride, 'same');
  const program = new Pool2DProgram(convInfo, 'max', positions);
  const res = NDArray.zeros(program.outputShape);
  const x = Array3D.randUniform(xShape, -1, 1);
  const binary = gpgpu_math.compileProgram(gpgpu, program, [x], res);

  const start = performance.now();
  for (let i = 0; i < OP_RUNS; i++) {
    gpgpu_math.runProgram(binary, [x], res);
  }
  res.getValues();
  const avgTime = (performance.now() - start) / OP_RUNS;

  x.dispose();
  res.dispose();
  texManager.dispose();
  gpgpu.deleteProgram(binary.webGLProgram);
  gpgpu.dispose();

  return avgTime;
}
