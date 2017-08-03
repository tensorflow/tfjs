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

export function getFragmentShaderSource(): string {
  return `
    precision highp float;
    uniform sampler2D matrixA;
    uniform sampler2D matrixB;
    uniform sampler2D matrixAScalar;
    uniform sampler2D matrixBScalar;
    varying vec2 resultUV;

    const vec2 halfTexel = vec2(0.5, 0.5);

    void main() {
      float a = texture2D(matrixA, resultUV).r;
      float b = texture2D(matrixB, resultUV).r;
      float aScalar = texture2D(matrixAScalar, halfTexel).r;
      float bScalar = texture2D(matrixBScalar, halfTexel).r;
      vec2 abScaled = vec2(a, b) * vec2(aScalar, bScalar);
      gl_FragColor = vec4(abScaled.x + abScaled.y, 0, 0, 0);
    }`;
}

export function addScaledMatrices(
    gpgpu: GPGPUContext, addScaledMatricesProgram: WebGLProgram,
    a: WebGLTexture, b: WebGLTexture, rows: number, columns: number,
    aScalar: WebGLTexture, bScalar: WebGLTexture, result: WebGLTexture) {
  gpgpu.setOutputMatrixTexture(result, rows, columns);
  gpgpu.setProgram(addScaledMatricesProgram);
  gpgpu.setInputMatrixTexture(a, 'matrixA', 0);
  gpgpu.setInputMatrixTexture(b, 'matrixB', 1);
  gpgpu.setInputMatrixTexture(aScalar, 'matrixAScalar', 2);
  gpgpu.setInputMatrixTexture(bScalar, 'matrixBScalar', 3);
  gpgpu.executeProgram();
}

export function uploadAddScaledMatricesDownload(
    a: Float32Array, b: Float32Array, rows: number, columns: number,
    aScalar: number, bScalar: number): Float32Array {
  const gpgpu = new GPGPUContext();
  const program: WebGLProgram = gpgpu.createProgram(getFragmentShaderSource());

  const aTex = gpgpu.createMatrixTexture(rows, columns);
  const bTex = gpgpu.createMatrixTexture(rows, columns);
  const aScalarTex = gpgpu.createMatrixTexture(1, 1);
  const bScalarTex = gpgpu.createMatrixTexture(1, 1);
  const resultTex = gpgpu.createMatrixTexture(rows, columns);

  gpgpu.uploadMatrixToTexture(aTex, rows, columns, a);
  gpgpu.uploadMatrixToTexture(bTex, rows, columns, b);
  gpgpu.uploadMatrixToTexture(aScalarTex, 1, 1, new Float32Array([aScalar]));
  gpgpu.uploadMatrixToTexture(bScalarTex, 1, 1, new Float32Array([bScalar]));

  addScaledMatrices(
      gpgpu, program, aTex, bTex, rows, columns, aScalarTex, bScalarTex,
      resultTex);

  const result = gpgpu.downloadMatrixFromTexture(resultTex, rows, columns);

  gpgpu.deleteMatrixTexture(aTex);
  gpgpu.deleteMatrixTexture(bTex);
  gpgpu.deleteMatrixTexture(resultTex);
  gpgpu.deleteMatrixTexture(aScalarTex);
  gpgpu.deleteMatrixTexture(bScalarTex);
  gpgpu.deleteProgram(program);
  gpgpu.dispose();

  return result;
}
