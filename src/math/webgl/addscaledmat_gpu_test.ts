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
import {Array1D, Array2D, initializeGPU, NDArray, Scalar} from '../ndarray';

import {AddScaledMatProgram} from './addscaledmat_gpu';
import {GPGPUContext} from './gpgpu_context';
import * as gpgpu_math from './gpgpu_math';
import {TextureManager} from './texture_manager';

function cpuAddScaledMatrices(
    a: Float32Array, aScalar: number, b: Float32Array,
    bScalar: number): Float32Array {
  const result = new Float32Array(a.length);
  for (let i = 0; i < result.length; ++i) {
    result[i] = (a[i] * aScalar) + (b[i] * bScalar);
  }
  return result;
}

describe('addscaledmat_gpu', () => {
  it('returns a matrix with the same shape as the input matrices', () => {
    const a = Array2D.zeros([9, 14]);
    const b = Array2D.zerosLike(a);
    const result = uploadAddScaledMatDownload(a, b, 0, 0);
    expect(result.length).toEqual(9 * 14);
  });

  it('returns A + B when scalars are 1', () => {
    const a = Array1D.new([1, 2, 3, 4, 5, 6]);
    const b = Array1D.new([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    const result = uploadAddScaledMatDownload(a, b, 1, 1);
    test_util.expectArraysClose(
        result, new Float32Array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6]));
  });

  it('returns A * aScalar when B and bScalar are 0', () => {
    const a = Array1D.new([1, 2, 3, 4, 5, 6]);
    const b = Array1D.zerosLike(a);
    const result = uploadAddScaledMatDownload(a, b, 1.1, 0);
    test_util.expectArraysClose(
        result, new Float32Array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6]));
  });

  it('returns B * bScalar when A and aScalar are 0', () => {
    const b = Array1D.new([1, 2, 3, 4, 5, 6]);
    const a = Array1D.zerosLike(b);
    const result = uploadAddScaledMatDownload(a, b, 0, 1.1);
    test_util.expectArraysClose(
        result, new Float32Array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6]));
  });

  it('returns (A * aScalar) + (B * bScalar)', () => {
    const a = Array2D.randUniform([12, 12], -2, 2);
    const aVals = a.getValues();
    const b = Array2D.randUniform([12, 12], -10, 10);
    const bVals = b.getValues();

    const c1 = 0.5;
    const c2 = 0.25;
    const result = uploadAddScaledMatDownload(a, b, c1, c2);
    test_util.expectArraysClose(
        result, cpuAddScaledMatrices(aVals, c1, bVals, c2));
  });
});

export function uploadAddScaledMatDownload(
    a: NDArray, b: NDArray, c1Val: number, c2Val: number): Float32Array {
  const c1 = Scalar.new(c1Val);
  const c2 = Scalar.new(c2Val);
  const gpgpu = new GPGPUContext();
  const textureManager = new TextureManager(gpgpu);
  initializeGPU(gpgpu, textureManager);

  const program = new AddScaledMatProgram(a.shape, b.shape);
  const res = NDArray.zeros(program.outputShape);
  const binary = gpgpu_math.compileProgram(gpgpu, program, [a, b, c1, c2], res);
  gpgpu_math.runProgram(binary, [a, b, c1, c2], res);

  const resValues = res.getValues();
  textureManager.dispose();
  gpgpu.deleteProgram(binary.webGLProgram);
  gpgpu.dispose();

  return resValues;
}
