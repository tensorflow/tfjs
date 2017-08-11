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
import * as conv_backprop_gpu from '../../src/math/webgl/conv_backprop_gpu';
import {GPGPUContext} from '../../src/math/webgl/gpgpu_context';
import * as test_util from '../../src/test_util';

import {BenchmarkTest} from './benchmark';

const OP_RUNS = 100;

export const BENCHMARK_TEST: BenchmarkTest = (size: number) => {
  const xShapeRCD: [number, number, number] = [size, size, 1];
  const origOutputDepth = 2;
  const fieldSize = 11;
  const origStride = 1;
  const origPad = 1;

  const gpgpu = new GPGPUContext();
  gpgpu.enableAutomaticDebugValidation(true);
  const origInputDepth = xShapeRCD[2];
  const src = conv_backprop_gpu.getFragmentShaderConvTransposeSource(
      xShapeRCD, fieldSize, origInputDepth, origStride, origPad, false);
  const program = gpgpu.createProgram(src);

  // Upload x.
  const xTexShapeRC = conv_util.computeTexShapeFrom3D(xShapeRCD);
  const xTex = gpgpu.createMatrixTexture(xTexShapeRC[0], xTexShapeRC[1]);
  const xData =
      test_util.randomArrayInRange(xTexShapeRC[0] * xTexShapeRC[1], -1, 1);
  gpgpu.uploadMatrixToTexture(xTex, xTexShapeRC[0], xTexShapeRC[1], xData);

  // Upload weights.
  const wTexShapeRC = conv_util.computeWeightsTexShape(
      origInputDepth, origOutputDepth, fieldSize);
  const wData =
      test_util.randomArrayInRange(wTexShapeRC[0] * wTexShapeRC[1], -1, 1);
  const wTex = gpgpu.createMatrixTexture(wTexShapeRC[0], wTexShapeRC[1]);
  gpgpu.uploadMatrixToTexture(wTex, wTexShapeRC[0], wTexShapeRC[1], wData);

  // Figure out the output shape by dilating the input.
  const dilatedRC =
      conv_util.computeDilatedRC([xShapeRCD[0], xShapeRCD[1]], origStride);
  const pad = fieldSize - 1 - origPad;
  const resultShapeRCD = conv_util.computeOutputShape3D(
      [dilatedRC[0], dilatedRC[1], origOutputDepth], fieldSize, origInputDepth,
      1, pad);

  const resultTexRC = conv_util.computeTexShapeFrom3D(resultShapeRCD);
  const resultTex = gpgpu.createMatrixTexture(resultTexRC[0], resultTexRC[1]);

  const start = performance.now();
  for (let i = 0; i < OP_RUNS; i++) {
    conv_backprop_gpu.convTranspose(
        gpgpu, program, xTex, wTex, null, resultTex, resultTexRC);
  }

  gpgpu.downloadMatrixFromTexture(resultTex, resultTexRC[0], resultTexRC[1]);

  const end = performance.now();

  const avgTime = (end - start) / OP_RUNS;

  gpgpu.deleteMatrixTexture(resultTex);
  gpgpu.deleteMatrixTexture(xTex);
  gpgpu.deleteMatrixTexture(wTex);
  gpgpu.deleteProgram(program);
  gpgpu.dispose();

  return avgTime;
};
