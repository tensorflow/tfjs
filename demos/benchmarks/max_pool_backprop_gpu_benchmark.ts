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
import {Array3D, initializeGPU, NDArray} from '../../src/math/ndarray';
import {GPGPUContext} from '../../src/math/webgl/gpgpu_context';
import * as gpgpu_math from '../../src/math/webgl/gpgpu_math';
// tslint:disable-next-line:max-line-length
import {MaxPool2DBackpropProgram} from '../../src/math/webgl/max_pool_backprop_gpu';
import {TextureManager} from '../../src/math/webgl/texture_manager';
import {BenchmarkTest} from './benchmark';

const OP_RUNS = 40;

export const BENCHMARK_TEST: BenchmarkTest = (size: number) => {
  const gpgpu = new GPGPUContext();
  const texManager = new TextureManager(gpgpu);
  initializeGPU(gpgpu, texManager);

  const depth = 1;
  const dyShape: [number, number, number] = [size, size, depth];
  const xShape: [number, number, number] = [size, size, depth];
  const fSize = 11;
  const stride = 1;
  const convInfo = conv_util.computeConvInfo(
      xShape, fSize, fSize, depth, stride, stride, 'same');
  const program = new MaxPool2DBackpropProgram(convInfo);
  const res = NDArray.zeros(program.outputShape);
  const dy = Array3D.randUniform(dyShape, -1, 1);
  const positionsData = new Float32Array(dy.size);
  for (let i = 0; i < positionsData.length; i++) {
    positionsData[i] = Math.floor(Math.random() * fSize * fSize);
  }
  const positions = Array3D.new(dyShape, positionsData);
  const binary =
      gpgpu_math.compileProgram(gpgpu, program, [dy, positions], res);

  const start = performance.now();
  for (let i = 0; i < OP_RUNS; i++) {
    gpgpu_math.runProgram(binary, [dy, positions], res);
  }
  res.getValues();
  const avgTime = (performance.now() - start) / OP_RUNS;

  dy.dispose();
  positions.dispose();
  res.dispose();
  texManager.dispose();
  gpgpu.deleteProgram(binary.webGLProgram);
  gpgpu.dispose();

  return avgTime;
};
