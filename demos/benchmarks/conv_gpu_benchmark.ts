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

import * as conv_util from '../../src/math/conv_util';
import {Array1D, Array3D, Array4D, initializeGPU} from '../../src/math/ndarray';
import {Conv2DProgram} from '../../src/math/webgl/conv_gpu';
import {GPGPUContext} from '../../src/math/webgl/gpgpu_context';
import * as gpgpu_math from '../../src/math/webgl/gpgpu_math';
import {TextureManager} from '../../src/math/webgl/texture_manager';

import {BenchmarkTest} from './benchmark';

const OP_RUNS = 40;

export const BENCHMARK_TEST: BenchmarkTest = (size: number) => {
  const gpgpu = new GPGPUContext();
  const texManager = new TextureManager(gpgpu);
  initializeGPU(gpgpu, texManager);

  const inDepth = 1;
  const inShape: [number, number, number] = [size, size, inDepth];
  const outDepth = 1;
  const filterSize = 11;
  const stride = 1;
  const hasBias = true;
  const convInfo = conv_util.computeConvInfo(
      inShape, filterSize, filterSize, outDepth, stride, stride, 'same');
  const program = new Conv2DProgram(convInfo, hasBias);
  const outputShape = program.outputShape as [number, number, number];
  const out = Array3D.zeros(outputShape);
  const x = Array3D.randUniform(inShape, -1, 1);
  const wShape =
      conv_util.computeWeightsShape4D(1, outDepth, filterSize, filterSize);
  const W = Array4D.randUniform(wShape, -1, 1);
  const b = Array1D.randUniform([outDepth], -1, 1);
  const inputs = [x, W, b];
  const binary = gpgpu_math.compileProgram(gpgpu, program, inputs, out);

  const start = performance.now();
  for (let i = 0; i < OP_RUNS; i++) {
    gpgpu_math.runProgram(binary, inputs, out);
  }
  out.getValues();
  const avgTime = (performance.now() - start) / OP_RUNS;

  x.dispose();
  W.dispose();
  b.dispose();
  out.dispose();
  texManager.dispose();
  gpgpu.deleteProgram(binary.webGLProgram);
  gpgpu.dispose();

  return avgTime;
};
