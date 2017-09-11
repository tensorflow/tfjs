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
import {ArgMinMaxProgram} from './argminmax_gpu';
import {GPGPUContext} from './gpgpu_context';
import * as gpgpu_math from './gpgpu_math';
import {TextureManager} from './texture_manager';
import {Array2D, Scalar, initializeGPU} from '../ndarray';

function uploadArgMinMaxDownload(
    a: Float32Array, rows: number, columns: number, op: 'min'|'max'): number {
  const arr = Array2D.new([rows, columns], a);
  const gpgpu = new GPGPUContext();
  const textureManager = new TextureManager(gpgpu);
  initializeGPU(gpgpu, textureManager);
  const out = Scalar.new(0);
  const program = new ArgMinMaxProgram(arr.size, op);
  const binary = gpgpu_math.compileProgram(gpgpu, program, [arr], out);
  gpgpu_math.runProgram(binary, [arr], out);

  const result = out.get();
  arr.dispose();
  textureManager.dispose();
  gpgpu.deleteProgram(binary.webGLProgram);
  gpgpu.dispose();
  return result;
}

function uploadArgMinDownload(
    a: Float32Array, rows: number, columns: number): number {
  return uploadArgMinMaxDownload(a, rows, columns, 'min');
}

function uploadArgMaxDownload(
    a: Float32Array, rows: number, columns: number): number {
  return uploadArgMinMaxDownload(a, rows, columns, 'max');
}

describe('argminmax_gpu ArgMin', () => {
  it('returns the only value in a 1x1 input matrix', () => {
    const a = new Float32Array([3]);
    const argMin = uploadArgMinDownload(a, 1, 1);
    expect(argMin).toEqual(0);
  });

  it('returns min indices when not in first cell', () => {
    const a = new Float32Array([0, 100, -12, 0]);  // row-major
    const argMin = uploadArgMinDownload(a, 2, 2);
    expect(argMin).toEqual(2);
  });

  it('finds the min value of a large array', () => {
    const a = new Float32Array(1024 * 1024);
    test_util.setValue(a, 1024, 1024, -100, 17, 913);
    const argMin = uploadArgMinDownload(a, 1024, 1024);
    expect(argMin).toEqual((17 * 1024) + 913);
  });

  it('returns the correct column and row when matrix is non-square', () => {
    const a = new Float32Array(19 * 254);
    test_util.setValue(a, 19, 254, -1, 13, 200);
    const argMin = uploadArgMinDownload(a, 19, 254);
    expect(argMin).toEqual((13 * 254) + 200);
  });

  it('works when the min element is the bottom/right cell in matrix', () => {
    const a = new Float32Array(129 * 129);
    test_util.setValue(a, 129, 129, -19, 128, 128);
    const argMin = uploadArgMinDownload(a, 129, 129);
    expect(argMin).toEqual((128 * 129) + 128);
  });
});

describe('argminmax_gpu ArgMax', () => {
  it('returns the only value in a 1x1 input matrix', () => {
    const a = new Float32Array([3]);
    const argMax = uploadArgMaxDownload(a, 1, 1);
    expect(argMax).toEqual(0);
  });

  it('returns min indices when not in first cell', () => {
    const a = new Float32Array([0, -12, 100, 0]);  // row-major
    const argMax = uploadArgMaxDownload(a, 2, 2);
    expect(argMax).toEqual(2);
  });

  it('finds the max value of a large array', () => {
    const a = new Float32Array(1024 * 1024);
    test_util.setValue(a, 1024, 1024, 100, 17, 913);
    const argMax = uploadArgMaxDownload(a, 1024, 1024);
    expect(argMax).toEqual((17 * 1024) + 913);
  });

  it('returns the correct column and row when matrix is non-square', () => {
    const a = new Float32Array(19 * 254);
    test_util.setValue(a, 19, 254, 109, 13, 200);
    const argMax = uploadArgMaxDownload(a, 19, 254);
    expect(argMax).toEqual((13 * 254) + 200);
  });

  it('works when the min element is the bottom/right cell in matrix', () => {
    const a = new Float32Array(129 * 129);
    test_util.setValue(a, 129, 129, 19, 128, 128);
    const argMax = uploadArgMaxDownload(a, 129, 129);
    expect(argMax).toEqual((128 * 129) + 128);
  });
});
