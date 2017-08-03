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
import * as max_pool_backprop_gpu from '../../src/math/webgl/max_pool_backprop_gpu';
import * as test_util from '../../src/test_util';
import * as util from '../../src/util';

import {BenchmarkTest} from './benchmark';

const OP_RUNS = 100;

export const BENCHMARK_TEST: BenchmarkTest = (size: number) => {
  const dyShapeRCD: [number, number, number] = [size, size, 1];
  const outputDepth = 1;
  const fieldSize = 11;
  const stride = 1;
  const zeroPad = conv_util.computeDefaultPad(dyShapeRCD, fieldSize, stride);
  const outputShapeRCD: [number, number, number] =
      conv_util.computeOutputShape3D(
          dyShapeRCD, fieldSize, outputDepth, stride, zeroPad);

  const dyTexShapeRC = conv_util.computeTexShapeFrom3D(dyShapeRCD);
  const outputTexShapeRC = conv_util.computeTexShapeFrom3D(outputShapeRCD);

  const gpgpu = new GPGPUContext();
  const program = gpgpu.createProgram(
      max_pool_backprop_gpu.getFragmentShaderMaxPoolBackprop(
          dyShapeRCD, fieldSize, stride, zeroPad));

  const dyTexture = gpgpu.createMatrixTexture(dyTexShapeRC[0], dyTexShapeRC[1]);
  const maxPositionsTexture =
      gpgpu.createMatrixTexture(dyTexShapeRC[0], dyTexShapeRC[1]);
  const outputTexture =
      gpgpu.createMatrixTexture(outputTexShapeRC[0], outputTexShapeRC[1]);

  const dyData =
      test_util.randomArrayInRange(dyTexShapeRC[0] * dyTexShapeRC[1], -1, 1);
  const maxPositionsData = new Float32Array(util.sizeFromShape(dyShapeRCD));
  for (let i = 0; i < maxPositionsData.length; i++) {
    maxPositionsData[i] = Math.floor(Math.random() * fieldSize * fieldSize);
  }

  gpgpu.uploadMatrixToTexture(
      dyTexture, dyTexShapeRC[0], dyTexShapeRC[1], dyData);
  gpgpu.uploadMatrixToTexture(
      maxPositionsTexture, dyTexShapeRC[0], dyTexShapeRC[1], maxPositionsData);

  const start = performance.now();
  for (let i = 0; i < OP_RUNS; i++) {
    max_pool_backprop_gpu.maxPoolBackprop(
        gpgpu, program, dyTexture, maxPositionsTexture, outputTexture,
        outputTexShapeRC);
  }

  gpgpu.downloadMatrixFromTexture(
      outputTexture, outputTexShapeRC[0], outputTexShapeRC[1]);
  const end = performance.now();

  const avgTime = (end - start) / OP_RUNS;

  gpgpu.deleteMatrixTexture(dyTexture);
  gpgpu.deleteMatrixTexture(maxPositionsTexture);
  gpgpu.deleteMatrixTexture(outputTexture);
  gpgpu.deleteProgram(program);
  gpgpu.dispose();

  return avgTime;
};