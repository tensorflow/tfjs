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

import * as argminmax_gpu from './argminmax_gpu';
import {GPGPUContext} from './gpgpu_context';
import {IS_NAN_SHADER_FUNC} from './webgl_util';

function getFragmentShaderPrologueSource(): string {
  return `
    precision highp float;
    uniform sampler2D matrixA;
    uniform sampler2D matrixB;
    varying vec2 resultUV;`;
}

function getFragmentShaderMainSource(): string {
  return `
    void main() {
      float argMaxA = getArgMinMax(matrixA);
      float argMaxB = getArgMinMax(matrixB);
      float value;
      if (isNaN(argMaxA)) {
        value = argMaxA;
      } else if (isNaN(argMaxB)) {
        value = argMaxB;
      } else {
        value = float(argMaxA == argMaxB);
      }
      gl_FragColor = vec4(value, 0, 0, 0);
    }`;
}

export function getArgMaxEqualsFragmentShaderSource(
    rows: number, columns: number): string {
  return [
    getFragmentShaderPrologueSource(),
    argminmax_gpu.getFragmentShaderGetArgMinMaxSource('>', rows, columns),
    getFragmentShaderMainSource()
  ].join('\n');
}

export function argMaxEquals(
    gpgpu: GPGPUContext, maxEqualsProgram: WebGLProgram, a: WebGLTexture,
    b: WebGLTexture, numRows: number, numCols: number, result: WebGLTexture) {
  gpgpu.setOutputMatrixTexture(result, 1, 1);
  gpgpu.setProgram(maxEqualsProgram);
  gpgpu.setInputMatrixTexture(a, 'matrixA', 0);
  gpgpu.setInputMatrixTexture(b, 'matrixB', 1);
  gpgpu.executeProgram();
}
