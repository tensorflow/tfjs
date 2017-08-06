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

export function getFragmentShaderPrologueSource(): string {
  return `
    precision highp float;
    uniform sampler2D x;
    uniform sampler2D weights;
    uniform sampler2D biases;
    varying vec2 resultUV;`;
}

export function getFragmentShaderGetMatrixValueOrZeroPadSource(): string {
  return `
    float getMatrixValueOrZeroPad(in sampler2D matrix, vec2 matrixShapeCR,
        vec2 requestedCR) {
      vec2 uv = (requestedCR + vec2(0.5, 0.5)) / matrixShapeCR;
      float value = texture2D(matrix, uv).r;
      bool lessThanZero = any(lessThan(uv, vec2(0, 0)));
      bool greaterThanOne = any(greaterThan(uv, vec2(1, 1)));
      bool outside = lessThanZero || greaterThanOne;
      return mix(value, 0.0, float(outside));
    }`;
}

export function getFragmentShaderConvolveSource(
    xShapeRCD: [number, number, number], fSize: number, outputDepth: number,
    stride: number, pad: number, hasBias: boolean) {
  const [xRows, xCols, inputDepth] = xShapeRCD;

  const xTexShapeRC = conv_util.computeTexShapeFrom3D(xShapeRCD);
  const wTexShapeRC =
      conv_util.computeWeightsTexShape(inputDepth, outputDepth, fSize);

  return `
    const vec2 halfCR = vec2(0.5, 0.5);
    const vec2 xShapeCR = vec2(${xTexShapeRC[1]}, ${xTexShapeRC[0]});
    const vec2 wShapeCR = vec2(${wTexShapeRC[1]}, ${wTexShapeRC[0]});

    void main() {
      vec2 yTexCR = floor(gl_FragCoord.xy);

      // Map from 2D (yTexR, yTexC) to 3D (yR, yC, d2).
      float yR = yTexCR.y;
      float yC = floor(yTexCR.x / ${outputDepth}.0);
      float d2 = mod(yTexCR.x, ${outputDepth}.0);
      float wTexC = d2;

      vec2 xRCCorner = vec2(yR, yC) * vec2(${stride}, ${stride}) -
          vec2(${pad}.0, ${pad}.0);
      float xRCorner = xRCCorner.x;
      float xCCorner = xRCCorner.y;

      // Convolve x(?, ?, d1) with w(:, :, d1, d2) to get y(yR, yC, d2).
      // ? = to be determined. : = across all values in that axis.
      float dotProd = 0.0;
      for (int wR = 0; wR < ${fSize}; wR++) {
        float wR_float = float(wR);
        float xR = xRCorner + wR_float;
        float xTexR = xR;

        for (int wC = 0; wC < ${fSize}; wC++) {
          float wC_float = float(wC);
          float xC = xCCorner + wC_float;

          for (int d1 = 0; d1 < ${inputDepth}; d1++) {
            float d1_float = float(d1);
            float xTexC = xC * ${inputDepth}.0 + d1_float;
            float wTexR = wR_float * ${fSize * inputDepth}.0 +
                wC_float * ${inputDepth}.0 + d1_float;

            float xValue =
                getMatrixValueOrZeroPad(x, xShapeCR, vec2(xTexC, xTexR));

            // Read w(wR, wC, d1, d2).
            vec2 wUV = (vec2(wTexC, wTexR) + halfCR) / wShapeCR;
            float wValue = texture2D(weights, wUV).r;

            dotProd += xValue * wValue;
          }
        }
      }
      if (${hasBias}) {
        dotProd += getBiasValue(biases, d2);
      }
      gl_FragColor = vec4(dotProd, 0, 0, 0);
    }`;
}

export function getFragmentShaderGetBiasValueSource(outputDepth: number):
    string {
  return `
    float getBiasValue(in sampler2D bias, float biasC) {
      const vec2 biasShapeCR = vec2(${outputDepth}, 1);
      vec2 biasCR = vec2(mod(biasC, ${outputDepth}.0), 0);
      vec2 biasUV = (biasCR + vec2(0.5, 0.5)) / biasShapeCR;
      return texture2D(bias, biasUV).r;
    }`;
}

export function getFragmentShaderSource(
    aShapeRowColDepth: [number, number, number], resultDepth: number,
    fieldSize: number, stride: number, zeroPad: number,
    hasBias: boolean): string {
  const aShapeRC: [number, number] =
      conv_util.computeTexShapeFrom3D(aShapeRowColDepth);

  const weightShapeRC: [number, number] = conv_util.computeWeightsTexShape(
      aShapeRowColDepth[2], resultDepth, fieldSize);

  const prologue = getFragmentShaderPrologueSource();
  const getMatrixValueOrZeroPad =
      getFragmentShaderGetMatrixValueOrZeroPadSource();
  const convolve = getFragmentShaderConvolveSource(
      aShapeRowColDepth, fieldSize, resultDepth, stride, zeroPad, hasBias);
  const getBiasValue = getFragmentShaderGetBiasValueSource(resultDepth);

  return [
    prologue,
    getMatrixValueOrZeroPad,
    getBiasValue,
    convolve,
  ].join('\n');
}

export function convolve(
    gpgpu: GPGPUContext, program: WebGLProgram, a: WebGLTexture,
    weights: WebGLTexture, biases: WebGLTexture|null, result: WebGLTexture,
    resultShapeRowCol: [number, number]) {
  gpgpu.setOutputMatrixTexture(
      result, resultShapeRowCol[0], resultShapeRowCol[1]);
  gpgpu.setProgram(program);
  gpgpu.setInputMatrixTexture(a, 'x', 0);
  gpgpu.setInputMatrixTexture(weights, 'weights', 1);
  if (biases != null) {
    gpgpu.setInputMatrixTexture(biases, 'biases', 2);
  }
  gpgpu.executeProgram();
}
