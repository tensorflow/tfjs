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

import * as conv_gpu from './conv_gpu';
import {GPGPUContext} from './gpgpu_context';

describe('conv_gpu getMatrixValueOrZeroPad', () => {
  function createGetMatrixValueOrZeroPadProgram(
      gpgpu: GPGPUContext, shapeRowCol: [number, number]): WebGLProgram {
    const prologue = conv_gpu.getFragmentShaderPrologueSource();

    const uniformColRow = 'uniform vec2 colRow;';

    const getMatrixValueOrZeroPad =
        conv_gpu.getFragmentShaderGetMatrixValueOrZeroPadSource();

    const main = `
        void main() {
          const vec2 aShapeCR = vec2(${shapeRowCol[1]}, ${shapeRowCol[0]});
          float value = getMatrixValueOrZeroPad(x, aShapeCR, colRow);
          gl_FragColor = vec4(value, 0, 0, 0);
        }`;

    const src =
        [prologue, uniformColRow, getMatrixValueOrZeroPad, main].join('\n');
    return gpgpu.createProgram(src);
  }

  function uploadGetMatrixValueOrZeroPadDownload(
      matrix: Float32Array, shapeRowCol: [number, number],
      paramRowCol: [number, number]): number {
    const gpgpu = new GPGPUContext();
    gpgpu.enableAutomaticDebugValidation(true);

    const program: WebGLProgram =
        createGetMatrixValueOrZeroPadProgram(gpgpu, shapeRowCol);

    const matrixTexture =
        gpgpu.createMatrixTexture(shapeRowCol[0], shapeRowCol[1]);
    const resultTexture = gpgpu.createMatrixTexture(1, 1);

    gpgpu.uploadMatrixToTexture(
        matrixTexture, shapeRowCol[0], shapeRowCol[1], matrix);

    gpgpu.setOutputMatrixTexture(resultTexture, 1, 1);
    gpgpu.setProgram(program);
    gpgpu.setInputMatrixTexture(matrixTexture, 'x', 0);
    const loc = gpgpu.getUniformLocation('colRow');
    gpgpu.gl.uniform2f(loc, paramRowCol[1], paramRowCol[0]);
    gpgpu.executeProgram();
    const result = gpgpu.downloadMatrixFromTexture(resultTexture, 1, 1);
    gpgpu.deleteMatrixTexture(resultTexture);
    gpgpu.deleteMatrixTexture(matrixTexture);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return result[0];
  }

  it('returns only value of a 1x1 matrix when row and column are 0', () => {
    const a = new Float32Array([1.23]);
    const result = uploadGetMatrixValueOrZeroPadDownload(a, [1, 1], [0, 0]);
    expect(result).toBeCloseTo(a[0]);
  });

  it('returns value of matrix cell at specified row and column', () => {
    const a = new Float32Array(32 * 64);
    a[5 + (30 * 64)] = Math.PI;
    const result = uploadGetMatrixValueOrZeroPadDownload(a, [32, 64], [30, 5]);
    expect(result).toBeCloseTo(Math.PI);
  });

  it('returns zero if sampling out-of-bounds left', () => {
    const a = new Float32Array(4 * 4);
    a.fill(1);
    const result = uploadGetMatrixValueOrZeroPadDownload(a, [4, 4], [0, -1]);
    expect(result).toEqual(0);
  });

  it('returns zero if sampling out-of-bounds right', () => {
    const a = new Float32Array(4 * 4);
    a.fill(1);
    const result = uploadGetMatrixValueOrZeroPadDownload(a, [4, 4], [0, 15]);
    expect(result).toEqual(0);
  });

  it('returns zero if sampling out-of-bounds top', () => {
    const a = new Float32Array(19 * 35);
    a.fill(1);
    const result = uploadGetMatrixValueOrZeroPadDownload(a, [19, 35], [-1, 0]);
    expect(result).toEqual(0);
  });

  it('returns zero if sampling out-of-bounds bottom', () => {
    const a = new Float32Array(19 * 35);
    a.fill(1);
    const result = uploadGetMatrixValueOrZeroPadDownload(a, [19, 35], [20, 0]);
    expect(result).toEqual(0);
  });

  it('returns zero if sampling out-of-bounds upper-left', () => {
    const a = new Float32Array(19 * 35);
    a.fill(1);
    const result = uploadGetMatrixValueOrZeroPadDownload(a, [19, 35], [-1, -1]);
    expect(result).toEqual(0);
  });

  it('returns zero if sampling out-of-bounds upper-right', () => {
    const a = new Float32Array(19 * 35);
    a.fill(1);
    const result = uploadGetMatrixValueOrZeroPadDownload(a, [19, 35], [-1, 36]);
    expect(result).toEqual(0);
  });

  it('returns zero if sampling out-of-bounds lower-left', () => {
    const a = new Float32Array(19 * 35);
    a.fill(1);
    const result = uploadGetMatrixValueOrZeroPadDownload(a, [19, 35], [20, -1]);
    expect(result).toEqual(0);
  });

  it('returns zero if sampling out-of-bounds lower-right', () => {
    const a = new Float32Array(19 * 35);
    a.fill(1);
    const result = uploadGetMatrixValueOrZeroPadDownload(a, [19, 35], [20, 36]);
    expect(result).toEqual(0);
  });
});
