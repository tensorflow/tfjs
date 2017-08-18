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

import {Array2D, initializeGPU, Scalar} from '../../src/math/ndarray';
import {GPGPUContext} from '../../src/math/webgl/gpgpu_context';
import * as gpgpu_math from '../../src/math/webgl/gpgpu_math';
import {LogSumExpProgram} from '../../src/math/webgl/logsumexp_gpu';
import {TextureManager} from '../../src/math/webgl/texture_manager';

import {BenchmarkTest} from './benchmark';

const OP_RUNS = 2;

export const BENCHMARK_TEST: BenchmarkTest = (size: number) => {
  const gpgpu = new GPGPUContext();
  const texManager = new TextureManager(gpgpu);
  initializeGPU(gpgpu, texManager);
  const out = new Scalar({texture: texManager.acquireTexture([1, 1])});
  const a = Array2D.randUniform([size, size], -1, 1);
  const program = new LogSumExpProgram(a.size);
  const binary = gpgpu_math.compileProgram(gpgpu, program, [a], out);

  const start = performance.now();
  for (let i = 0; i < OP_RUNS; i++) {
    gpgpu_math.runProgram(binary, [a], out);
  }
  out.getValues();
  const avgTime = (performance.now() - start) / OP_RUNS;
  a.dispose();
  out.dispose();
  texManager.dispose();
  gpgpu.deleteProgram(binary.webGLProgram);
  gpgpu.dispose();

  return avgTime;
};
