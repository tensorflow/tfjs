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

export function getFragmentShaderPrologueSource(): string {
  return `
    precision highp float;
    uniform sampler2D matrixA;
    varying vec2 resultUV;`;
}

function getFragmentShaderMainSource(): string {
  return `
    void main() {
      gl_FragColor = vec4(getArgMinMax(matrixA), 0, 0, 0);
    }`;
}

function getArgMinMaxFragmentShaderSource(
    rows: number, columns: number, compOp: string): string {
  return [
    getFragmentShaderPrologueSource(),
    getFragmentShaderGetArgMinMaxSource(compOp, rows, columns),
    getFragmentShaderMainSource()
  ].join('\n');
}

export function getArgMinFragmentShaderSource(
    rows: number, columns: number): string {
  return getArgMinMaxFragmentShaderSource(rows, columns, '<');
}

export function getArgMaxFragmentShaderSource(
    rows: number, columns: number): string {
  return getArgMinMaxFragmentShaderSource(rows, columns, '>');
}

export function getFragmentShaderGetArgMinMaxSource(
    compOp: string, rows: number, columns: number) {
  return `
    const vec2 dimCR = vec2(${columns}.0, ${rows}.0);
    const vec2 halfCR = vec2(0.5, 0.5);

    ${IS_NAN_SHADER_FUNC}

    float getArgMinMax(in sampler2D matrix) {
      vec2 bestCR = vec2(0, 0);
      float bestValue = texture2D(matrix, bestCR).r;

      for (int c = 0; c < ${columns}; c++) {
        for (int r = 0; r < ${rows}; r++) {
          vec2 cr = vec2(c, r);
          vec2 uv = (cr + halfCR) / dimCR;
          float value = texture2D(matrix, uv).r;
          if (isNaN(value)) {
            return value;
          }
          if (value ${compOp} bestValue) {
            bestValue = value;
            bestCR = cr;
          }
        }
      }
      return bestCR.x + (bestCR.y * dimCR.x);
    }
  `;
}

export function argMinMax(
    gpgpu: GPGPUContext, minMaxProgram: WebGLProgram, a: WebGLTexture,
    aNumRows: number, aNumCols: number, result: WebGLTexture) {
  gpgpu.setOutputMatrixTexture(result, 1, 1);
  gpgpu.setProgram(minMaxProgram);
  gpgpu.setInputMatrixTexture(a, 'matrixA', 0);
  gpgpu.executeProgram();
}
