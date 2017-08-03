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
    aResultUV: string, bResultUV: string, op: string): string {
  return `
    precision highp float;
    uniform sampler2D matrixA;
    uniform sampler2D matrixB;
    varying vec2 resultUV;

    void main() {
      float a = texture2D(matrixA, ${aResultUV}).r;
      float b = texture2D(matrixB, ${bResultUV}).r;
      ${op}
    }`;
}

export function binaryOp(
    gpgpu: GPGPUContext, program: WebGLProgram, a: WebGLTexture,
    aShapeRowCol: [number, number], b: WebGLTexture,
    bShapeRowCol: [number, number], result: WebGLTexture,
    resultShapeRowCol: [number, number]) {
  gpgpu.setOutputMatrixTexture(
      result, resultShapeRowCol[0], resultShapeRowCol[1]);
  gpgpu.setProgram(program);
  gpgpu.setInputMatrixTexture(a, 'matrixA', 0);
  gpgpu.setInputMatrixTexture(b, 'matrixB', 1);
  gpgpu.executeProgram();
}

export function uploadBinaryOpDownload(
    a: Float32Array, aShape: [number, number], b: Float32Array,
    bShape: [number, number], fragmentShaderSource: string): Float32Array {
  const gpgpu = new GPGPUContext();
  const program = gpgpu.createProgram(fragmentShaderSource);

  const aTexture: WebGLTexture =
      gpgpu.createMatrixTexture(aShape[0], aShape[1]);
  const bTexture: WebGLTexture =
      gpgpu.createMatrixTexture(bShape[0], bShape[1]);

  const resultShape: [number, number] =
      [Math.max(aShape[0], bShape[0]), Math.max(aShape[1], bShape[1])];

  const resultTexture: WebGLTexture =
      gpgpu.createMatrixTexture(resultShape[0], resultShape[1]);

  gpgpu.uploadMatrixToTexture(aTexture, aShape[0], aShape[1], a);
  gpgpu.uploadMatrixToTexture(bTexture, bShape[0], bShape[1], b);

  binaryOp(
      gpgpu, program, aTexture, aShape, bTexture, bShape, resultTexture,
      resultShape);
  const result = gpgpu.downloadMatrixFromTexture(
      resultTexture, resultShape[0], resultShape[1]);

  gpgpu.deleteMatrixTexture(aTexture);
  gpgpu.deleteMatrixTexture(bTexture);
  gpgpu.deleteMatrixTexture(resultTexture);
  gpgpu.deleteProgram(program);
  gpgpu.dispose();
  return result;
}
