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

export function getFragmentShaderSource(
    aNumRows: number, aNumCols: number, bNumRows: number, bNumCols: number,
    resultNumRows: number, resultNumCols: number): string {
  return `
    precision highp float;
    uniform sampler2D matrixA;
    uniform sampler2D matrixB;
    varying vec2 resultUV;

    const vec2 aDimCR = vec2(${aNumCols}.0, ${aNumRows}.0);
    const vec2 bDimCR = vec2(${bNumCols}.0, ${bNumRows}.0);
    const vec2 resultDimCR = vec2(${resultNumCols}.0, ${resultNumRows}.0);
    const vec4 halfCR = vec4(0.5, 0.5, 0.5, 0.5);

    void main() {
      vec2 resultCR = floor(resultUV * resultDimCR);
      vec4 resultCRBroadcast = vec4(resultCR, resultCR);
      vec4 abDimsCR = vec4(aDimCR, bDimCR);
      vec4 abCR = mod(resultCRBroadcast, abDimsCR);
      vec4 abCRCenters = abCR + halfCR;
      vec4 abUV = abCRCenters / abDimsCR;
      vec4 a = texture2D(matrixA, abUV.rg);
      vec4 b = texture2D(matrixB, abUV.ba);
      float product = a.r * b.r;
      gl_FragColor = vec4(product, 0, 0, 0);
    }`;
}

export function multiplyBroadcast(
    gpgpu: GPGPUContext, multiplyBroadcastProgram: WebGLProgram,
    a: WebGLTexture, aNumRows: number, aNumCols: number, b: WebGLTexture,
    bNumRows: number, bNumCols: number, result: WebGLTexture,
    resultNumRows: number, resultNumCols: number) {
  gpgpu.setOutputMatrixTexture(result, resultNumRows, resultNumCols);
  gpgpu.setProgram(multiplyBroadcastProgram);
  gpgpu.setInputMatrixTexture(a, 'matrixA', 0);
  gpgpu.setInputMatrixTexture(b, 'matrixB', 1);
  gpgpu.executeProgram();
}

export function uploadMultiplyBroadcastDownload(
    a: Float32Array, aNumRows: number, aNumCols: number, b: Float32Array,
    bNumRows: number, bNumCols: number): Float32Array {
  const resultNumRows = Math.max(aNumRows, bNumRows);
  const resultNumCols = Math.max(aNumCols, bNumCols);

  const gpgpu = new GPGPUContext();
  const program: WebGLProgram = gpgpu.createProgram(getFragmentShaderSource(
      aNumRows, aNumCols, bNumRows, bNumCols, resultNumRows, resultNumCols));

  const aTexture: WebGLTexture = gpgpu.createMatrixTexture(aNumRows, aNumCols);
  const bTexture: WebGLTexture = gpgpu.createMatrixTexture(bNumRows, bNumCols);
  const resultTexture: WebGLTexture =
      gpgpu.createMatrixTexture(resultNumRows, resultNumCols);

  gpgpu.uploadMatrixToTexture(aTexture, aNumRows, aNumCols, a);
  gpgpu.uploadMatrixToTexture(bTexture, bNumRows, bNumCols, b);

  multiplyBroadcast(
      gpgpu, program, aTexture, aNumRows, aNumCols, bTexture, bNumRows,
      bNumCols, resultTexture, resultNumRows, resultNumCols);

  const result = gpgpu.downloadMatrixFromTexture(
      resultTexture, resultNumRows, resultNumCols);

  gpgpu.deleteMatrixTexture(aTexture);
  gpgpu.deleteMatrixTexture(bTexture);
  gpgpu.deleteMatrixTexture(resultTexture);
  gpgpu.deleteProgram(program);
  gpgpu.dispose();

  return result;
}
