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

import * as conv_util from '../../src/math/conv_util';
import {GPGPUContext} from '../../src/math/webgl/gpgpu_context';
import * as max_pool_gpu from '../../src/math/webgl/max_pool_gpu';
import * as test_util from '../../src/test_util';

import {BenchmarkTest} from './benchmark';

const OP_RUNS = 40;

export const MAX_POOL_BENCHMARK_TEST: BenchmarkTest = (size: number) => {
  const inputShapeRCD: [number, number, number] = [size, size, 1];
  const outputDepth = 1;
  const fieldSize = 11;
  const stride = 1;
  const zeroPad = conv_util.computeDefaultPad(inputShapeRCD, fieldSize, stride);
  const outputShapeRCD: [number, number, number] =
      conv_util.computeOutputShape3D(
          inputShapeRCD, fieldSize, outputDepth, stride, zeroPad);

  const inputTexShapeRC = conv_util.computeTexShapeFrom3D(inputShapeRCD);
  const outputTexShapeRC = conv_util.computeTexShapeFrom3D(outputShapeRCD);

  const gpgpu = new GPGPUContext();
  const program =
      gpgpu.createProgram(max_pool_gpu.getFragmentShaderMaxPoolSource(
          inputShapeRCD, fieldSize, stride, zeroPad));

  const inputTexture =
      gpgpu.createMatrixTexture(inputTexShapeRC[0], inputTexShapeRC[1]);
  const outputTexture =
      gpgpu.createMatrixTexture(outputTexShapeRC[0], outputTexShapeRC[1]);

  const inputData = test_util.randomArrayInRange(
      inputTexShapeRC[0] * inputTexShapeRC[1], -1, 1);

  gpgpu.uploadMatrixToTexture(
      inputTexture, inputTexShapeRC[0], inputTexShapeRC[1], inputData);

  const start = performance.now();
  for (let i = 0; i < OP_RUNS; i++) {
    max_pool_gpu.maxPoolCommon(
        gpgpu, program, inputTexture, outputTexture, outputTexShapeRC);
  }

  gpgpu.downloadMatrixFromTexture(
      outputTexture, outputTexShapeRC[0], outputTexShapeRC[1]);
  const end = performance.now();

  const avgTime = (end - start) / OP_RUNS;

  gpgpu.deleteMatrixTexture(inputTexture);
  gpgpu.deleteMatrixTexture(outputTexture);
  gpgpu.deleteProgram(program);
  gpgpu.dispose();

  return avgTime;
};

export const MAX_POOL_POSNS_BENCHMARK_TEST: BenchmarkTest = (size: number) => {
  const inputShapeRCD: [number, number, number] = [size, size, 1];
  const outputDepth = 1;
  const fieldSize = 11;
  const stride = 1;
  const zeroPad = conv_util.computeDefaultPad(inputShapeRCD, fieldSize, stride);
  const outputShapeRCD: [number, number, number] =
      conv_util.computeOutputShape3D(
          inputShapeRCD, fieldSize, outputDepth, stride, zeroPad);

  const inputTexShapeRC = conv_util.computeTexShapeFrom3D(inputShapeRCD);
  const outputTexShapeRC = conv_util.computeTexShapeFrom3D(outputShapeRCD);

  const gpgpu = new GPGPUContext();
  const program: WebGLProgram =
      gpgpu.createProgram(max_pool_gpu.getFragmentShaderMaxPoolPositionsSource(
          inputShapeRCD, fieldSize, stride, zeroPad));

  const inputTexture =
      gpgpu.createMatrixTexture(inputTexShapeRC[0], inputTexShapeRC[1]);
  const outputTexture =
      gpgpu.createMatrixTexture(outputTexShapeRC[0], outputTexShapeRC[1]);

  const inputData = test_util.randomArrayInRange(
      inputTexShapeRC[0] * inputTexShapeRC[1], -1, 1);

  gpgpu.uploadMatrixToTexture(
      inputTexture, inputTexShapeRC[0], inputTexShapeRC[1], inputData);

  const start = performance.now();
  for (let i = 0; i < OP_RUNS; i++) {
    max_pool_gpu.maxPoolCommon(
        gpgpu, program, inputTexture, outputTexture, outputTexShapeRC);
  }

  gpgpu.downloadMatrixFromTexture(
      outputTexture, outputTexShapeRC[0], outputTexShapeRC[1]);
  const end = performance.now();

  const avgTime = (end - start) / OP_RUNS;

  gpgpu.deleteMatrixTexture(inputTexture);
  gpgpu.deleteMatrixTexture(outputTexture);
  gpgpu.deleteProgram(program);
  gpgpu.dispose();

  return avgTime;
};