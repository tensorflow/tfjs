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

import * as util from '../../util';
import {GPGPUContext} from './gpgpu_context';

export function getFragmentShaderSource(): string {
  return `
    precision highp float;
    uniform sampler2D matrixA;
    uniform vec2 inputDimCR;
    uniform vec2 resultDimCR;
    varying vec2 resultUV;
    const vec2 halfCR = vec2(0.5, 0.5);

    void main() {
      vec2 resultCR = floor(resultUV * resultDimCR);
      // indexInFlat = row * stride + column, where stride == numOutputColumns
      float indexInFlat = resultCR.y * resultDimCR.x + resultCR.x;

      vec2 inputCR = vec2(
        mod(indexInFlat, inputDimCR.x), // col = indexInFlat % numInputColumns
        floor(indexInFlat / inputDimCR.x) // row = indexInFlat / numInputColumns
      ) + halfCR;

      vec2 inputUV = inputCR / inputDimCR;
      gl_FragColor = texture2D(matrixA, inputUV);
    }`;
}

export function reshape(
    gpgpu: GPGPUContext, reshapeProgram: WebGLProgram, a: WebGLTexture,
    aNumRows: number, aNumCols: number, result: WebGLTexture,
    resultNumRows: number, resultNumCols: number) {
  const inputSize = aNumRows * aNumCols;
  const outputSize = resultNumCols * resultNumRows;
  util.assert(
      inputSize === outputSize,
      `The input size (${inputSize}) and output size (${outputSize}) ` +
          `must match`);

  gpgpu.setOutputMatrixTexture(result, resultNumRows, resultNumCols);
  gpgpu.setProgram(reshapeProgram);
  gpgpu.setInputMatrixTexture(a, 'matrixA', 0);

  const inputDimCRLocation = gpgpu.getUniformLocation('inputDimCR');
  gpgpu.gl.uniform2f(inputDimCRLocation, aNumCols, aNumRows);

  const resultDimCRLocation = gpgpu.getUniformLocation('resultDimCR');
  gpgpu.gl.uniform2f(resultDimCRLocation, resultNumCols, resultNumRows);

  gpgpu.executeProgram();
}
