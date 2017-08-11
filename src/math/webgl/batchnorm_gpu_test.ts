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

import * as test_util from '../../test_util';

import * as batchnorm_gpu from './batchnorm_gpu';
import {GPGPUContext} from './gpgpu_context';

describe('batchnorm gpu test', () => {
  function uploadBatchNormDownload(
      x: Float32Array, xTexShapeRowCol: [number, number], mean: Float32Array,
      meanTexShapeRowCol: [number, number], variance: Float32Array,
      varianceTexShapeRowCol: [number, number], offset: Float32Array|null,
      offsetTexShapeRowCol: [number, number]|null, scale: Float32Array|null,
      scaleTexShapeRowCol: [number, number]|null,
      varianceEpsilon: number): Float32Array {
    const resultTexShapeRC: [number, number] = xTexShapeRowCol;
    const gpgpu = new GPGPUContext();
    gpgpu.enableAutomaticDebugValidation(true);

    const shaderSource = batchnorm_gpu.getFragmentShaderSource(
        xTexShapeRowCol, meanTexShapeRowCol, varianceTexShapeRowCol,
        offsetTexShapeRowCol, scaleTexShapeRowCol, varianceEpsilon);

    const program = gpgpu.createProgram(shaderSource);

    const xTex =
        gpgpu.createMatrixTexture(xTexShapeRowCol[0], xTexShapeRowCol[1]);
    const meanTex =
        gpgpu.createMatrixTexture(meanTexShapeRowCol[0], meanTexShapeRowCol[1]);
    const varianceTex = gpgpu.createMatrixTexture(
        varianceTexShapeRowCol[0], varianceTexShapeRowCol[1]);

    let offsetTex = null;
    if (offset != null) {
      offsetTex = gpgpu.createMatrixTexture(
          offsetTexShapeRowCol![0], offsetTexShapeRowCol![1]);
    }
    let scaleTex = null;
    if (scale != null) {
      scaleTex = gpgpu.createMatrixTexture(
          scaleTexShapeRowCol![0], scaleTexShapeRowCol![1]);
    }

    const resultTex =
        gpgpu.createMatrixTexture(resultTexShapeRC[0], resultTexShapeRC[1]);

    gpgpu.uploadMatrixToTexture(
        xTex, xTexShapeRowCol[0], xTexShapeRowCol[1], x);
    gpgpu.uploadMatrixToTexture(
        meanTex, meanTexShapeRowCol[0], meanTexShapeRowCol[1], mean);
    gpgpu.uploadMatrixToTexture(
        varianceTex, varianceTexShapeRowCol[0], varianceTexShapeRowCol[1],
        variance);
    if (offset != null) {
      gpgpu.uploadMatrixToTexture(
          offsetTex!, offsetTexShapeRowCol![0], offsetTexShapeRowCol![1],
          offset);
    }
    if (scale != null) {
      gpgpu.uploadMatrixToTexture(
          scaleTex!, scaleTexShapeRowCol![0], scaleTexShapeRowCol![1], scale);
    }

    batchnorm_gpu.batchNormalization(
        gpgpu, program, xTex, xTexShapeRowCol, meanTex, meanTexShapeRowCol,
        varianceTex, varianceTexShapeRowCol, offsetTex, offsetTexShapeRowCol,
        scaleTex, scaleTexShapeRowCol, resultTex, resultTexShapeRC);

    const result = gpgpu.downloadMatrixFromTexture(
        resultTex, resultTexShapeRC[0], resultTexShapeRC[1]);

    gpgpu.deleteMatrixTexture(resultTex);
    gpgpu.deleteMatrixTexture(xTex);
    gpgpu.deleteMatrixTexture(meanTex);
    gpgpu.deleteMatrixTexture(varianceTex);
    if (offsetTex != null) {
      gpgpu.deleteMatrixTexture(offsetTex);
    }
    if (scaleTex != null) {
      gpgpu.deleteMatrixTexture(scaleTex);
    }
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return result;
  }

  it('simple batchnorm, no offset or scale, 2x1x2', () => {
    const x = new Float32Array([2, 100, 4, 400]);
    const mean = new Float32Array([1, 2]);
    const variance = new Float32Array([2, 3]);
    const varianceEpsilon = .001;

    const result = uploadBatchNormDownload(
        x, [2, 2], mean, [1, 2], variance, [1, 2], null, null, null, null,
        varianceEpsilon);

    const expectedResult = new Float32Array([
      (x[0] - mean[0]) * 1 / Math.sqrt(variance[0] + varianceEpsilon),
      (x[1] - mean[1]) * 1 / Math.sqrt(variance[1] + varianceEpsilon),
      (x[2] - mean[0]) * 1 / Math.sqrt(variance[0] + varianceEpsilon),
      (x[3] - mean[1]) * 1 / Math.sqrt(variance[1] + varianceEpsilon)
    ]);
    test_util.expectArraysClose(result, expectedResult, 1e-5);
  });

  it('simple batchnorm, no offset, 2x1x2', () => {
    const x = new Float32Array([2, 100, 4, 400]);
    const mean = new Float32Array([1, 2]);
    const variance = new Float32Array([2, 3]);
    const scale = new Float32Array([4, 5]);
    const varianceEpsilon = .001;

    const result = uploadBatchNormDownload(
        x, [2, 2], mean, [1, 2], variance, [1, 2], null, null, scale, [1, 2],
        varianceEpsilon);

    const expectedResult = new Float32Array([
      (x[0] - mean[0]) * scale[0] / Math.sqrt(variance[0] + varianceEpsilon),
      (x[1] - mean[1]) * scale[1] / Math.sqrt(variance[1] + varianceEpsilon),
      (x[2] - mean[0]) * scale[0] / Math.sqrt(variance[0] + varianceEpsilon),
      (x[3] - mean[1]) * scale[1] / Math.sqrt(variance[1] + varianceEpsilon)
    ]);
    test_util.expectArraysClose(result, expectedResult, 1e-5);
  });

  it('simple batchnorm, no scale, 2x1x2', () => {
    const x = new Float32Array([2, 100, 4, 400]);
    const mean = new Float32Array([1, 2]);
    const variance = new Float32Array([2, 3]);
    const offset = new Float32Array([4, 5]);
    const varianceEpsilon = .001;

    const result = uploadBatchNormDownload(
        x, [2, 2], mean, [1, 2], variance, [1, 2], offset, [1, 2], null, null,
        varianceEpsilon);

    const expectedResult = new Float32Array([
      offset[0] +
          (x[0] - mean[0]) * 1 / Math.sqrt(variance[0] + varianceEpsilon),
      offset[1] +
          (x[1] - mean[1]) * 1 / Math.sqrt(variance[1] + varianceEpsilon),
      offset[0] +
          (x[2] - mean[0]) * 1 / Math.sqrt(variance[0] + varianceEpsilon),
      offset[1] +
          (x[3] - mean[1]) * 1 / Math.sqrt(variance[1] + varianceEpsilon)
    ]);
    test_util.expectArraysClose(result, expectedResult, 1e-5);
  });

  it('simple batchnorm, 2x1x2', () => {
    const x = new Float32Array([2, 100, 4, 400]);
    const mean = new Float32Array([1, 2]);
    const variance = new Float32Array([2, 3]);
    const offset = new Float32Array([3, 4]);
    const scale = new Float32Array([4, 5]);
    const varianceEpsilon = .001;

    const result = uploadBatchNormDownload(
        x, [2, 2], mean, [1, 2], variance, [1, 2], offset, [1, 2], scale,
        [1, 2], varianceEpsilon);

    const expectedResult = new Float32Array([
      offset[0] +
          (x[0] - mean[0]) * scale[0] /
              Math.sqrt(variance[0] + varianceEpsilon),
      offset[1] +
          (x[1] - mean[1]) * scale[1] /
              Math.sqrt(variance[1] + varianceEpsilon),
      offset[0] +
          (x[2] - mean[0]) * scale[0] /
              Math.sqrt(variance[0] + varianceEpsilon),
      offset[1] +
          (x[3] - mean[1]) * scale[1] / Math.sqrt(variance[1] + varianceEpsilon)
    ]);
    test_util.expectArraysClose(result, expectedResult, 1e-5);
  });

  it('batchnorm matches tensorflow, 2x3x3', () => {
    const x = new Float32Array([
      0.49955603, 0.04158615, -1.09440524, 2.03854165, -0.61578344, 2.87533573,
      1.18105987, 0.807462, 1.87888837, 2.26563962, -0.37040935, 1.35848753,
      -0.75347094, 0.15683117, 0.91925946, 0.34121279, 0.92717143, 1.89683965
    ]);
    const mean = new Float32Array([0.39745062, -0.48062894, 0.4847822]);
    const variance = new Float32Array([0.32375343, 0.67117643, 1.08334653]);
    const offset = new Float32Array([0.69398749, -1.29056387, 0.9429723]);
    const scale = new Float32Array([-0.5607271, 0.9878457, 0.25181573]);
    const varianceEpsilon = .001;

    const result = uploadBatchNormDownload(
        x, [2, 9], mean, [1, 3], variance, [1, 3], offset, [1, 3], scale,
        [1, 3], varianceEpsilon);

    const expectedResult = new Float32Array([
      0.59352049, -0.66135202, 0.5610874, -0.92077015, -1.45341019, 1.52106473,
      -0.07704776, 0.26144429, 1.28010017, -1.14422404, -1.15776136, 1.15425493,
      1.82644104, -0.52249442, 1.04803919, 0.74932291, 0.40568101, 1.2844412
    ]);
    test_util.expectArraysClose(result, expectedResult, 1e-5);
  });
});
