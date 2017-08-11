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

import * as conv_util from '../conv_util';

import {GPGPUContext} from './gpgpu_context';

export function getFragmentShaderSource(
    inputShapeRCD: [number, number, number],
    outputDimensionsRowCol: [number, number], alignCorners: boolean): string {
  const depth = inputShapeRCD[2];

  const inputTexShapeRC = conv_util.computeTexShapeFrom3D(inputShapeRCD);

  const effectiveInputShapeRCD = alignCorners ?
      [inputShapeRCD[0] - 1, inputShapeRCD[1] - 1, depth] :
      inputShapeRCD;

  const effectiveOutputShapeRCD = alignCorners ?
      [outputDimensionsRowCol[0] - 1, outputDimensionsRowCol[1] - 1, depth] :
      [outputDimensionsRowCol[0], outputDimensionsRowCol[1], depth];

  return `
    precision highp float;
    uniform sampler2D matrixA;
    varying vec2 resultUV;
    const vec2 halfCR = vec2(0.5, 0.5);

    const vec2 inputShapeCR = vec2(${inputShapeRCD[1]}, ${inputShapeRCD[0]});
    const vec2 inputShapeTexCR = vec2(
        ${inputTexShapeRC[1]}, ${inputTexShapeRC[0]});

    const vec2 effectiveInputOverOutputRatioCR = vec2(
        ${effectiveInputShapeRCD[1] / effectiveOutputShapeRCD[1]},
        ${effectiveInputShapeRCD[0] / effectiveOutputShapeRCD[0]});

    float sampleInput(float col, float row, float d) {
      vec2 uv = (vec2(col * ${depth}.0 + d, row) + halfCR) / inputShapeTexCR;
      return texture2D(matrixA, uv).r;
    }

    void main() {
      vec2 yTexCR = floor(gl_FragCoord.xy);

      // Map from 2D (yTexR, yTexC) to 3D (yR, yC, d).
      vec2 yCR = vec2(floor(yTexCR.x / ${depth}.0), yTexCR.y);
      float d = mod(yTexCR.x, ${depth}.0);

      // Fractional source index.
      vec2 sourceFracIndexCR = yCR * effectiveInputOverOutputRatioCR;

      // Compute the four integer indices.
      vec2 sourceFloorCR = floor(sourceFracIndexCR);
      vec2 sourceCeilCR = min(inputShapeCR - 1.0, ceil(sourceFracIndexCR));

      float topLeft = sampleInput(sourceFloorCR[0], sourceFloorCR[1], d);
      float bottomLeft = sampleInput(sourceFloorCR[0], sourceCeilCR[1], d);
      float topRight = sampleInput(sourceCeilCR[0], sourceFloorCR[1], d);
      float bottomRight = sampleInput(sourceCeilCR[0], sourceCeilCR[1], d);

      vec2 fracCR = sourceFracIndexCR - sourceFloorCR;

      float top = topLeft + (topRight - topLeft) * fracCR[0];
      float bottom = bottomLeft + (bottomRight - bottomLeft) * fracCR[0];
      float newValue = top + (bottom - top) * fracCR[1];

      gl_FragColor = vec4(newValue, 0.0, 0.0, 0.0);
    }`;
}

export function resizeBilinear(
    gpgpu: GPGPUContext, resizeBilinearProgram: WebGLProgram, a: WebGLTexture,
    result: WebGLTexture, resultShapeRowCol: [number, number]) {
  gpgpu.setOutputMatrixTexture(
      result, resultShapeRowCol[0], resultShapeRowCol[1]);
  gpgpu.setProgram(resizeBilinearProgram);
  gpgpu.setInputMatrixTexture(a, 'matrixA', 0);
  gpgpu.executeProgram();
}
