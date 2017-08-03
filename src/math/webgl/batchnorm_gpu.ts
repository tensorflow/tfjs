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
    xTexShapeRC: [number, number], meanTexShapeRC: [number, number],
    varianceTexShapeRC: [number, number],
    offsetTexShapeRC: [number, number]|null,
    scaleTexShapeRC?: [number, number]|null, varianceEpsilon = 0.001): string {
  let offsetSamplerSnippet = '';
  let offsetShapeInitializationSnippet = '';
  let offsetCoordsSnippet = '';
  let offsetUVSnippet = '';
  let offsetValueSnippet = '';
  let offsetOperationSnippet = '0.0';

  let scaleSamplerSnippet = '';
  let scaleShapeInitializationSnippet = '';
  let scaleCoordsSnippet = '';
  let scaleUVSnippet = '';
  let scaleValueSnippet = '';
  let scaleOperationSnippet = '';

  if (offsetTexShapeRC != null) {
    offsetSamplerSnippet = 'uniform sampler2D offset;';
    offsetShapeInitializationSnippet = `const vec2 offsetShapeCR = vec2(
            ${offsetTexShapeRC[1]}, ${offsetTexShapeRC[0]});`;
    offsetCoordsSnippet = 'vec2 offsetCoordsCR = mod(yTexCR, offsetShapeCR);';
    offsetUVSnippet =
        'vec2 offsetUV = (offsetCoordsCR + halfCR) / offsetShapeCR;';
    offsetValueSnippet = 'float offsetValue = texture2D(offset, offsetUV).r;';
    offsetOperationSnippet = 'offsetValue';
  }

  if (scaleTexShapeRC != null) {
    scaleSamplerSnippet = 'uniform sampler2D scale;';
    scaleShapeInitializationSnippet = `const vec2 scaleShapeCR = vec2(
            ${scaleTexShapeRC[1]}, ${scaleTexShapeRC[0]});`;
    scaleCoordsSnippet = 'vec2 scaleCoordsCR = mod(yTexCR, scaleShapeCR);';
    scaleUVSnippet = 'vec2 scaleUV = (scaleCoordsCR + halfCR) / scaleShapeCR;';
    scaleValueSnippet = 'float scaleValue = texture2D(scale, scaleUV).r;';
    scaleOperationSnippet = 'inv *= scaleValue;';
  }

  return `
    precision highp float;
    uniform sampler2D x;
    uniform sampler2D mean;
    uniform sampler2D variance;
    ${offsetSamplerSnippet}
    ${scaleSamplerSnippet}

    varying vec2 resultUV;

    const vec2 xShapeCR = vec2(${xTexShapeRC[1]}, ${xTexShapeRC[0]});
    const vec2 meanShapeCR = vec2(${meanTexShapeRC[1]}, ${meanTexShapeRC[0]});
    const vec2 varianceShapeCR = vec2(
        ${varianceTexShapeRC[1]}, ${varianceTexShapeRC[0]});

    ${offsetShapeInitializationSnippet}
    ${scaleShapeInitializationSnippet}

    const vec2 halfCR = vec2(0.5, 0.5);
    const float varianceEpsilon = ${varianceEpsilon};

    void main() {
      vec2 yTexCR = floor(gl_FragCoord.xy);

      vec2 meanCoordsCR = mod(yTexCR, meanShapeCR);
      vec2 varianceCoordsCR = mod(yTexCR, varianceShapeCR);
      ${offsetCoordsSnippet}
      ${scaleCoordsSnippet}

      vec2 meanUV = (meanCoordsCR + halfCR) / meanShapeCR;
      vec2 varianceUV = (varianceCoordsCR + halfCR) / varianceShapeCR;
      ${offsetUVSnippet}
      ${scaleUVSnippet}

      float xValue = texture2D(x, resultUV).r;
      float meanValue = texture2D(mean, meanUV).r;
      float varianceValue = texture2D(variance, varianceUV).r;
      ${offsetValueSnippet}
      ${scaleValueSnippet}

      float inv = 1.0 / sqrt(varianceValue + varianceEpsilon);
      ${scaleOperationSnippet}
      float xTimesInv = xValue * inv;
      float meanTimesInvWithOffset = ${offsetOperationSnippet}
          - meanValue * inv;

      gl_FragColor = vec4(xTimesInv + meanTimesInvWithOffset, 0, 0, 0);
    }`;
}

export function batchNormalization(
    gpgpu: GPGPUContext, program: WebGLProgram, x: WebGLTexture,
    xShapeRowCol: [number, number], mean: WebGLTexture,
    meanShapeRowCol: [number, number], variance: WebGLTexture,
    varianceShapeRowCol: [number, number], offset: WebGLTexture|null,
    offsetShapeRowCol: [number, number]|null, scale: WebGLTexture|null,
    scaleShapeRowCol: [number, number]|null, result: WebGLTexture,
    resultShapeRowCol: [number, number]) {
  gpgpu.setOutputMatrixTexture(
      result, resultShapeRowCol[0], resultShapeRowCol[1]);
  gpgpu.setProgram(program);
  gpgpu.setInputMatrixTexture(x, 'x', 0);
  gpgpu.setInputMatrixTexture(mean, 'mean', 1);
  gpgpu.setInputMatrixTexture(variance, 'variance', 2);
  let nextIndex = 3;
  if (offset != null) {
    gpgpu.setInputMatrixTexture(offset, 'offset', nextIndex);
    nextIndex++;
  }
  if (scale != null) {
    gpgpu.setInputMatrixTexture(scale, 'scale', nextIndex);
  }
  gpgpu.executeProgram();
}