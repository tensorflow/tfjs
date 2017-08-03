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
import * as argmaxequals_gpu from './argmaxequals_gpu';
import {GPGPUContext} from './gpgpu_context';

function uploadArgMaxEqualsDownload(
    a: Float32Array, b: Float32Array, rows: number, columns: number): number {
  const src =
      argmaxequals_gpu.getArgMaxEqualsFragmentShaderSource(rows, columns);
  const gpgpu = new GPGPUContext();
  const program = gpgpu.createProgram(src);
  const aTexture = gpgpu.createMatrixTexture(rows, columns);
  const bTexture = gpgpu.createMatrixTexture(rows, columns);
  const resultTexture = gpgpu.createMatrixTexture(1, 1);
  gpgpu.uploadMatrixToTexture(aTexture, rows, columns, a);
  gpgpu.uploadMatrixToTexture(bTexture, rows, columns, b);
  argmaxequals_gpu.argMaxEquals(
      gpgpu, program, aTexture, bTexture, rows, columns, resultTexture);
  const result = new Float32Array(4);
  gpgpu.gl.readPixels(0, 0, 1, 1, gpgpu.gl.RGBA, gpgpu.gl.FLOAT, result);
  gpgpu.deleteMatrixTexture(aTexture);
  gpgpu.deleteMatrixTexture(resultTexture);
  gpgpu.deleteProgram(program);
  gpgpu.dispose();
  return result[0];
}

describe('argmaxequals_gpu ArgMin', () => {
  it('one value in each array', () => {
    const a = new Float32Array([3]);
    const b = new Float32Array([3]);
    const equals = uploadArgMaxEqualsDownload(a, b, 1, 1);
    expect(equals).toEqual(1);
  });

  it('different argmax values', () => {
    const a = new Float32Array([2, 3]);
    const b = new Float32Array([3, 2]);
    const equals = uploadArgMaxEqualsDownload(a, b, 1, 2);
    expect(equals).toEqual(0);
  });

  it('same argmax values', () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 4, 3, 2, 1]);
    const b = new Float32Array([10, 2, 30, 4, 50, 4, 30, 2, 10]);
    const equals = uploadArgMaxEqualsDownload(a, b, 1, 9);
    expect(equals).toEqual(1);
  });
});