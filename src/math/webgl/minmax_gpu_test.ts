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
import * as minmax_gpu from './minmax_gpu';

function uploadMinMaxDownloadDriver(
    a: Float32Array, rows: number, columns: number,
    fragmentShaderSource: string): Float32Array {
  const gpgpu = new GPGPUContext();
  const program: WebGLProgram = gpgpu.createProgram(fragmentShaderSource);
  const aTexture: WebGLTexture = gpgpu.createMatrixTexture(rows, columns);
  const resultTexture: WebGLTexture = gpgpu.createMatrixTexture(1, 1);
  gpgpu.uploadMatrixToTexture(aTexture, rows, columns, a);
  minmax_gpu.minMax(gpgpu, program, aTexture, rows, columns, resultTexture);
  const result = new Float32Array(4);
  gpgpu.gl.readPixels(0, 0, 1, 1, gpgpu.gl.RGBA, gpgpu.gl.FLOAT, result);
  gpgpu.deleteMatrixTexture(resultTexture);
  gpgpu.deleteMatrixTexture(aTexture);
  gpgpu.deleteProgram(program);
  gpgpu.dispose();
  return result;
}

function uploadMinDownload(
    a: Float32Array, rows: number, columns: number): number {
  const src = minmax_gpu.getMinFragmentShaderSource(rows, columns);
  const result = uploadMinMaxDownloadDriver(a, rows, columns, src);
  return result[0];
}

function uploadMaxDownload(
    a: Float32Array, rows: number, columns: number): number {
  const src = minmax_gpu.getMaxFragmentShaderSource(rows, columns);
  const result = uploadMinMaxDownloadDriver(a, rows, columns, src);
  return result[0];
}

describe('minmax_gpu min', () => {
  it('returns the only value in a 1x1 input matrix', () => {
    const a = new Float32Array([3.141]);
    const minValue = uploadMinDownload(a, 1, 1);
    expect(minValue).toEqual(a[0]);
  });

  it('returns min value from the first cell of a 2x1', () => {
    const a = new Float32Array([-100, 100]);
    const minValue = uploadMinDownload(a, 2, 1);
    expect(minValue).toEqual(a[0]);
  });

  it('returns min value from the second cell of a 2x1', () => {
    const a = new Float32Array([100, -1.234]);
    const minValue = uploadMinDownload(a, 2, 1);
    expect(minValue).toEqual(a[1]);
  });

  it('finds the min value of a large array', () => {
    const a = new Float32Array(1024 * 1024);
    a[a.length - 91] = -0.1;
    const minValue = uploadMinDownload(a, 1024, 1024);
    expect(minValue).toBeCloseTo(-0.1);
  });
});

describe('minmax_gpu max', () => {
  it('returns the only value in a 1x1 input matrix', () => {
    const a = new Float32Array([3.141]);
    const maxValue = uploadMaxDownload(a, 1, 1);
    expect(maxValue).toEqual(a[0]);
  });

  it('returns max value from the first cell of a 2x1', () => {
    const a = new Float32Array([100, -100]);
    const maxValue = uploadMaxDownload(a, 2, 1);
    expect(maxValue).toEqual(a[0]);
  });

  it('returns max value from the second cell of a 2x1', () => {
    const a = new Float32Array([-1.234, 100]);
    const maxValue = uploadMaxDownload(a, 2, 1);
    expect(maxValue).toEqual(a[1]);
  });

  it('finds the max value of a large array', () => {
    const a = new Float32Array(1024 * 1024);
    a[a.length - 91] = 0.1;
    const maxValue = uploadMaxDownload(a, 1024, 1024);
    expect(maxValue).toBeCloseTo(0.1);
  });
});
