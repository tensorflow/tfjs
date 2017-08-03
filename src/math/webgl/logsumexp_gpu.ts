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

export function getFragmentShaderSource(rows: number, columns: number): string {
  return `
    precision highp float;
    uniform sampler2D matrixA;
    varying vec2 resultUV;

    const vec2 aDimCR = vec2(${columns}.0, ${rows}.0);
    const vec2 halfCR = vec2(0.5, 0.5);

    void main() {
      float aMax = texture2D(matrixA, halfCR / aDimCR).r;
      for (float r = 0.0; r < aDimCR.y; r += 1.0) {
        for (float c = 0.0; c < aDimCR.x; c += 1.0) {
          vec2 uv = (vec2(c, r) + halfCR) / aDimCR;
          float aCur = texture2D(matrixA, uv).r;
          aMax = max(aMax, aCur);
        }
      }

      float expSum = 0.0;
      for (float r = 0.0; r < aDimCR.y; r += 1.0) {
        for (float c = 0.0; c < aDimCR.x; c += 1.0) {
          vec2 uv = (vec2(c, r) + halfCR) / aDimCR;
          float aCur = texture2D(matrixA, uv).r;
          expSum += exp(aCur - aMax);
        }
      }

      gl_FragColor = vec4(aMax + log(expSum), 0, 0, 0);
    }`;
}

export function logSumExp(
    gpgpu: GPGPUContext, logSumExpProgram: WebGLProgram, a: WebGLTexture,
    rows: number, columns: number, result: WebGLTexture) {
  gpgpu.setOutputMatrixTexture(result, 1, 1);
  gpgpu.setProgram(logSumExpProgram);
  gpgpu.setInputMatrixTexture(a, 'matrixA', 0);
  gpgpu.executeProgram();
}

export function uploadLogSumExpDownload(
    a: Float32Array, rows: number, columns: number): number {
  const gpgpu = new GPGPUContext();
  const program = gpgpu.createProgram(getFragmentShaderSource(rows, columns));
  const aTexture = gpgpu.createMatrixTexture(rows, columns);
  const resultTexture = gpgpu.createMatrixTexture(1, 1);
  gpgpu.uploadMatrixToTexture(aTexture, rows, columns, a);
  logSumExp(gpgpu, program, aTexture, rows, columns, resultTexture);
  const result = gpgpu.downloadMatrixFromTexture(resultTexture, 1, 1);
  gpgpu.deleteMatrixTexture(aTexture);
  gpgpu.deleteMatrixTexture(resultTexture);
  gpgpu.deleteProgram(program);
  gpgpu.dispose();
  return result[0];
}
