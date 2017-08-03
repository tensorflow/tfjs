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

export function getFragmentShaderSource(resultOp: string): string {
  return `
    precision highp float;
    uniform sampler2D matrixA;
    varying vec2 resultUV;

    void main() {
      float value = texture2D(matrixA, resultUV).r;
      ${resultOp}
    }`;
}

export function unaryOp(
    gpgpu: GPGPUContext, unaryOpProgram: WebGLProgram, a: WebGLTexture,
    rows: number, columns: number, result: WebGLTexture) {
  gpgpu.setOutputMatrixTexture(result, rows, columns);
  gpgpu.setProgram(unaryOpProgram);
  gpgpu.setInputMatrixTexture(a, 'matrixA', 0);
  gpgpu.executeProgram();
}

export function uploadUnaryOpDownload(
    a: Float32Array, rows: number, columns: number,
    resultOp: string): Float32Array {
  const gpgpu = new GPGPUContext();
  const fragmentShaderSrc = getFragmentShaderSource(resultOp);
  const program: WebGLProgram = gpgpu.createProgram(fragmentShaderSrc);
  const aTexture: WebGLTexture = gpgpu.createMatrixTexture(rows, columns);
  const resultTexture: WebGLTexture = gpgpu.createMatrixTexture(rows, columns);
  gpgpu.uploadMatrixToTexture(aTexture, rows, columns, a);
  unaryOp(gpgpu, program, aTexture, rows, columns, resultTexture);
  const result = gpgpu.downloadMatrixFromTexture(resultTexture, rows, columns);
  gpgpu.deleteMatrixTexture(aTexture);
  gpgpu.deleteMatrixTexture(resultTexture);
  gpgpu.deleteProgram(program);
  gpgpu.dispose();
  return result;
}
