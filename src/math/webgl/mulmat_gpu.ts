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

import {MatrixOrientation} from '../math';
import {Array2D} from '../ndarray';

import {GPGPUContext} from './gpgpu_context';
import * as shader_compiler from './shader_compiler';

export function getFragmentShader(
    a: Array2D, b: Array2D, out: Array2D, aOrientation: MatrixOrientation,
    bOrientation: MatrixOrientation): string {
  const sharedDim =
      (aOrientation === MatrixOrientation.REGULAR ? a.shape[1] : a.shape[0]);
  const aSnippet =
      (aOrientation === MatrixOrientation.REGULAR) ? 'aRow, i' : 'i, aRow';
  const bSnippet =
      (bOrientation === MatrixOrientation.REGULAR) ? 'i, bCol' : 'bCol, i';

  const inputs = [{name: 'matrixA', array: a}, {name: 'matrixB', array: b}];
  const userCode = `
    const float sharedDim = ${sharedDim}.0;

    float dotARowBCol(float aRow, float bCol) {
      float result = 0.0;
      for (float i = 0.0; i < sharedDim; i += 1.0) {
        float a = getMatrixA(${aSnippet});
        float b = getMatrixB(${bSnippet});
        result += (a * b);
      }
      return result;
    }

    void main() {
      vec2 resRC = getOutputCoords();
      setOutput(dotARowBCol(resRC.x, resRC.y));
    }
  `;
  return shader_compiler.makeShader(inputs, out, userCode);
}

export function multiplyMatrix(
    gpgpu: GPGPUContext, multiplyProgram: WebGLProgram, a: WebGLTexture,
    b: WebGLTexture, result: WebGLTexture, outTexShape: [number, number]) {
  gpgpu.setOutputMatrixTexture(result, outTexShape[0], outTexShape[1]);
  gpgpu.setProgram(multiplyProgram);
  gpgpu.setInputMatrixTexture(a, 'matrixA', 0);
  gpgpu.setInputMatrixTexture(b, 'matrixB', 1);
  gpgpu.executeProgram();
}
