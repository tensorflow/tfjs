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
import * as conv_gpu from '../../src/math/webgl/conv_gpu';
import {GPGPUContext} from '../../src/math/webgl/gpgpu_context';
import * as test_util from '../../src/test_util';

import {BenchmarkTest} from './benchmark';

const OP_RUNS = 40;

export const BENCHMARK_TEST: BenchmarkTest = (size: number) => {
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
  const weightsTexShapeRC = conv_util.computeWeightsTexShape(
      inputShapeRCD[2], outputDepth, fieldSize);
  const biasesTexShapeRC = conv_util.computeBiasesTexShape(outputDepth);

  const hasBias = true;
  const gpgpu = new GPGPUContext();
  const program = gpgpu.createProgram(conv_gpu.getFragmentShaderSource(
      inputShapeRCD, outputDepth, fieldSize, stride, zeroPad, hasBias));

  const inputTexture =
      gpgpu.createMatrixTexture(inputTexShapeRC[0], inputTexShapeRC[1]);
  const weightsTexture =
      gpgpu.createMatrixTexture(weightsTexShapeRC[0], weightsTexShapeRC[1]);
  const biasesTexture =
      gpgpu.createMatrixTexture(biasesTexShapeRC[0], biasesTexShapeRC[1]);
  const outputTexture =
      gpgpu.createMatrixTexture(outputTexShapeRC[0], outputTexShapeRC[1]);

  const inputData = test_util.randomArrayInRange(
      inputTexShapeRC[0] * inputTexShapeRC[1], -1, 1);
  const weightsData = test_util.randomArrayInRange(
      weightsTexShapeRC[0] * weightsTexShapeRC[1], -1, 1);
  const biasesData = test_util.randomArrayInRange(
      biasesTexShapeRC[0] * biasesTexShapeRC[1], -1, 1);

  gpgpu.uploadMatrixToTexture(
      inputTexture, inputTexShapeRC[0], inputTexShapeRC[1], inputData);
  gpgpu.uploadMatrixToTexture(
      weightsTexture, weightsTexShapeRC[0], weightsTexShapeRC[1], weightsData);
  gpgpu.uploadMatrixToTexture(
      biasesTexture, biasesTexShapeRC[0], biasesTexShapeRC[1], biasesData);

  const start = performance.now();
  for (let i = 0; i < OP_RUNS; i++) {
    conv_gpu.convolve(
        gpgpu, program, inputTexture, weightsTexture, biasesTexture,
        outputTexture, outputTexShapeRC);
  }

  gpgpu.downloadMatrixFromTexture(
      outputTexture, outputTexShapeRC[0], outputTexShapeRC[1]);
  const end = performance.now();

  const avgTime = (end - start) / OP_RUNS;

  gpgpu.deleteMatrixTexture(inputTexture);
  gpgpu.deleteMatrixTexture(weightsTexture);
  gpgpu.deleteMatrixTexture(biasesTexture);
  gpgpu.deleteMatrixTexture(outputTexture);
  gpgpu.deleteProgram(program);
  gpgpu.dispose();

  return avgTime;
};
