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
import {IS_NAN_SHADER_FUNC} from './webgl_util';

function getFragmentShaderSource(
    rows: number, columns: number, compOp: string): string {
  return `
    precision highp float;
    uniform sampler2D matrixA;
    varying vec2 outputColumnRow;

    const vec2 aDimCR = vec2(${columns}.0, ${rows}.0);
    const vec2 halfCR = vec2(0.5, 0.5);

    ${IS_NAN_SHADER_FUNC}

    void main() {
      float value = texture2D(matrixA, halfCR / aDimCR).r;
      for (int r = 0; r < ${rows}; r++) {
        for (int c = 0; c < ${columns}; c++) {
          vec2 cr = vec2(c, r);
          vec2 uv = (cr + halfCR) / aDimCR;
          float candidate = texture2D(matrixA, uv).r;
          if (isNaN(candidate)) {
            gl_FragColor = vec4(candidate, 0, 0, 0);
            return;
          }
          value = ${compOp}(value, candidate);
        }
      }
      gl_FragColor = vec4(value, 0, 0, 0);
    }`;
}

export function getMinFragmentShaderSource(
    rows: number, columns: number): string {
  return getFragmentShaderSource(rows, columns, 'min');
}

export function getMaxFragmentShaderSource(
    rows: number, columns: number): string {
  return getFragmentShaderSource(rows, columns, 'max');
}

export function minMax(
    gpgpu: GPGPUContext, minMaxProgram: WebGLProgram, a: WebGLTexture,
    rows: number, columns: number, result: WebGLTexture) {
  gpgpu.setOutputMatrixTexture(result, 1, 1);
  gpgpu.setProgram(minMaxProgram);
  gpgpu.setInputMatrixTexture(a, 'matrixA', 0);
  gpgpu.executeProgram();
}
