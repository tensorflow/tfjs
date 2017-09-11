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

import * as test_util from '../../test_util';
import {initializeGPU, NDArray} from '../ndarray';

import {BatchNormProgram} from './batchnorm_gpu';
import {GPGPUContext} from './gpgpu_context';
import * as gpgpu_math from './gpgpu_math';
import {TextureManager} from './texture_manager';

describe('batchnorm gpu test', () => {

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
        x, [2, 3, 3], mean, [1, 3], variance, [1, 3], offset, [1, 3], scale,
        [1, 3], varianceEpsilon);

    const expectedResult = new Float32Array([
      0.59352049, -0.66135202, 0.5610874, -0.92077015, -1.45341019, 1.52106473,
      -0.07704776, 0.26144429, 1.28010017, -1.14422404, -1.15776136, 1.15425493,
      1.82644104, -0.52249442, 1.04803919, 0.74932291, 0.40568101, 1.2844412
    ]);
    test_util.expectArraysClose(result, expectedResult, 1e-5);
  });
});

function uploadBatchNormDownload(
    x: Float32Array, xShape: number[], mean: Float32Array, meanShape: number[],
    variance: Float32Array, varianceShape: number[], offset: Float32Array|null,
    offsetShape: number[]|null, scale: Float32Array|null,
    scaleShape: number[]|null, varianceEpsilon: number): Float32Array {
  const gpgpu = new GPGPUContext();
  const textureManager = new TextureManager(gpgpu);
  initializeGPU(gpgpu, textureManager);

  const program = new BatchNormProgram(
      xShape, meanShape, varianceShape, offsetShape, scaleShape,
      varianceEpsilon);
  const xArr = NDArray.make(xShape, {values: x});
  const meanArr = NDArray.make(meanShape, {values: mean});
  const varianceArr = NDArray.make(varianceShape, {values: variance});
  const inputs = [xArr, meanArr, varianceArr];

  if (offset != null) {
    const offsetArr = NDArray.make(offsetShape, {values: offset});
    inputs.push(offsetArr);
  }
  if (scale != null) {
    const scaleArr = NDArray.make(scaleShape, {values: scale});
    inputs.push(scaleArr);
  }

  const res = NDArray.zeros(program.outputShape);
  const binary = gpgpu_math.compileProgram(gpgpu, program, inputs, res);
  gpgpu_math.runProgram(binary, inputs, res);
  const resValues = res.getValues();

  textureManager.dispose();
  gpgpu.deleteProgram(binary.webGLProgram);
  gpgpu.dispose();

  return resValues;
}
