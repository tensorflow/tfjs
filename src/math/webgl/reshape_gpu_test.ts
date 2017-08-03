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

import {GPGPUContext} from './gpgpu_context';
import * as reshape_gpu from './reshape_gpu';

describe('reshape_gpu', () => {
  let gpgpu: GPGPUContext;

  beforeEach(() => {
    gpgpu = new GPGPUContext();
    gpgpu.enableAutomaticDebugValidation(true);
  });

  afterEach(() => {
    gpgpu.dispose();
  });

  it('reshape a 2x3 matrix into the same size', () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const result = uploadReshapeDownload(a, 2, 3, 2, 3);
    expect(result).toEqual(a);
  });

  it('reshape a 2x3 matrix into a column (6x1)', () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const result = uploadReshapeDownload(a, 2, 3, 6, 1);
    expect(result).toEqual(a);
  });

  it('reshape a 2x3 matrix into a row (1x6) vector', () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const result = uploadReshapeDownload(a, 2, 3, 1, 6);
    expect(result).toEqual(a);
  });

  it('reshape a 2x3 into a 3x2 matrix', () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const result = uploadReshapeDownload(a, 2, 3, 3, 2);
    expect(result).toEqual(a);
  });

  it('reshape a 2x3 into a 3x1 causes an error', () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const f = () => {
      uploadReshapeDownload(a, 2, 3, 3, 1);
    };

    expect(f).toThrowError();
  });

  function uploadReshapeDownload(
      a: Float32Array, aNumRows: number, aNumCols: number,
      resultNumRows: number, resultNumCols: number): Float32Array {
    const program = gpgpu.createProgram(reshape_gpu.getFragmentShaderSource());

    const aTexture = gpgpu.createMatrixTexture(aNumRows, aNumCols);
    gpgpu.uploadMatrixToTexture(aTexture, aNumRows, aNumCols, a);

    const resultTexture: WebGLTexture =
        gpgpu.createMatrixTexture(resultNumRows, resultNumCols);

    reshape_gpu.reshape(
        gpgpu, program, aTexture, aNumRows, aNumCols, resultTexture,
        resultNumRows, resultNumCols);

    const result = gpgpu.downloadMatrixFromTexture(
        resultTexture, resultNumRows, resultNumCols);

    gpgpu.deleteMatrixTexture(aTexture);
    gpgpu.deleteMatrixTexture(resultTexture);
    gpgpu.deleteProgram(program);

    return result;
  }
});
