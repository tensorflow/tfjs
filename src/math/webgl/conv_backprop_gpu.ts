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

import * as conv_gpu from './conv_gpu';
import {GPGPUContext} from './gpgpu_context';

export function getFragmentShaderDerWeightsSource(
    xShapeRowColDepth: [number, number, number], fSize: number,
    outputDepth: number, stride: number, zeroPad: number) {
  const getMatrixValueOrZeroPad =
      conv_gpu.getFragmentShaderGetMatrixValueOrZeroPadSource();
  const inputDepth = xShapeRowColDepth[2];

  const xTexShapeRC = conv_util.computeTexShapeFrom3D(xShapeRowColDepth);

  const yShape = conv_util.computeOutputShape3D(
      xShapeRowColDepth, fSize, outputDepth, stride, zeroPad);
  const yNumRows = yShape[0];
  const yNumCols = yShape[1];
  const yTexShapeRC = conv_util.computeTexShapeFrom3D(yShape);

  const fSizeTimesInputDepth = fSize * inputDepth;

  const prologue = `
    precision highp float;
    uniform sampler2D x;
    uniform sampler2D dy;
  `;

  return prologue + '\n' + getMatrixValueOrZeroPad + '\n' +
      `
    const vec2 halfCR = vec2(0.5, 0.5);
    const vec2 xShapeCR = vec2(${xTexShapeRC[1]}, ${xTexShapeRC[0]});
    const vec2 dyShapeCR = vec2(${yTexShapeRC[1]}, ${yTexShapeRC[0]});

    void main() {
      vec2 wTexCR = floor(gl_FragCoord.xy);

      // Map from 2D (wTexR, wTexC) to 4D (wR, wC, d1, d2).
      float wR = floor(wTexCR.y / ${fSizeTimesInputDepth}.0);
      float wTexRLeftover = wTexCR.y - wR * ${fSizeTimesInputDepth}.0;
      float wC = floor(wTexRLeftover / ${inputDepth}.0);
      float d1 = mod(wTexRLeftover, ${inputDepth}.0);
      float d2 = wTexCR.x;

      // Convolve x(?, ?, d1) with dy(:, :, d2) to get dw(wR, wC, d1, d2).
      // ? = to be determined. : = across all values in that axis.
      float dotProd = 0.0;
      for (float yR = 0.0; yR < ${yNumRows}.0; yR += 1.0) {
        float xR = wR + yR * ${stride}.0 - ${zeroPad}.0;
        float xTexR = xR;
        float yTexR = yR;
        for (float yC = 0.0; yC < ${yNumCols}.0; yC += 1.0) {
          float xC = wC + yC * ${stride}.0 - ${zeroPad}.0;

          // Map from 3D (xR, xC, d1) to 2D (xTexR, xTexC).
          // Map from 3D (yR, yC, d2) to 2D (yTexR, yTexC).
          vec2 xyTexC = vec2(xC, yC) * vec2(${inputDepth}.0, ${outputDepth}.0) +
                        vec2(d1, d2);
          float xTexC = xyTexC.x;
          float yTexC = xyTexC.y;

          // Read dy(yR, yC, d2).
          vec2 dyUV = (vec2(yTexC, yTexR) + halfCR) / dyShapeCR;
          float dyValue = texture2D(dy, dyUV).r;

          // Read x(xR, xC, d1) (potentially zero-padded).
          float xValue =
            getMatrixValueOrZeroPad(x, xShapeCR, vec2(xTexC, xTexR));

          dotProd += (xValue * dyValue);
        }
      }
      gl_FragColor = vec4(dotProd, 0, 0, 0);
    }`;
}

export function getFragmentShaderConvTransposeSource(
    xShapeRCD: [number, number, number], fSize: number, origInputDepth: number,
    origStride: number, origPad: number, hasBias: boolean) {
  const pad = fSize - 1 - origPad;
  const [xRows, xCols, origOutputDepth] = xShapeRCD;

  const xTexShapeRC = conv_util.computeTexShapeFrom3D(xShapeRCD);
  const wTexShapeRC =
      conv_util.computeWeightsTexShape(origInputDepth, origOutputDepth, fSize);

  const getBiasValue = hasBias ?
      conv_gpu.getFragmentShaderGetBiasValueSource(origInputDepth) :
      '';
  const biasPrologue = hasBias ? 'uniform sampler2D biases;' : '';
  const biasOperation = hasBias ? 'dotProd += getBiasValue(biases, d2);' : '';

  const prologue = `
    precision highp float;
    uniform sampler2D x;
    uniform sampler2D weights;
    ${biasPrologue}
    `;

  return prologue + '\n' + getBiasValue + '\n' +
      `
    const vec2 halfCR = vec2(0.5, 0.5);
    const vec2 xShapeCR = vec2(${xTexShapeRC[1]}, ${xTexShapeRC[0]});
    const vec2 wShapeCR = vec2(${wTexShapeRC[1]}, ${wTexShapeRC[0]});

    void main() {
      vec2 yTexCR = floor(gl_FragCoord.xy);

      // Map from 2D (yTexR, yTexC) to 3D (yR, yC, d2).
      float yR = yTexCR.y;
      float yC = floor(yTexCR.x / ${origInputDepth}.0);
      float d2 = mod(yTexCR.x, ${origInputDepth}.0);

      vec2 xRCCorner = vec2(yR, yC) - vec2(${pad}.0, ${pad}.0);
      float xRCorner = xRCCorner.x;
      float xCCorner = xRCCorner.y;

      // Convolve x(?, ?, d1) with w(:, :, d2, d1) to get y(yR, yC, d2).
      // ? = to be determined. : = across all values in that axis.
      float dotProd = 0.0;
      for (float wR = 0.0; wR < ${fSize}.0; wR += 1.0) {

        float xR = (xRCorner + wR) / ${origStride}.0;
        // TODO(smilkov): Splice this with another version where you call
        // getMatrixValueOrZeroPad(). Here and below.
        if (xR < 0.0 || xR >= ${xRows}.0 || fract(xR) > 0.0) {
          continue;
        }

        float wRPerm = ${fSize}.0 - 1.0 - wR;
        float xTexR = xR;

        for (float wC = 0.0; wC < ${fSize}.0; wC += 1.0) {

          float xC = (xCCorner + wC) / ${origStride}.0;
          if (xC < 0.0 || xC >= ${xCols}.0 || fract(xC) > 0.0) {
            continue;
          }

          float wCPerm = ${fSize}.0 - 1.0 - wC;
          float wTexR = wRPerm * ${fSize}.0 * ${origInputDepth}.0 +
                        wCPerm * ${origInputDepth}.0 + d2;

          for (float d1 = 0.0; d1 < ${origOutputDepth}.0; d1 += 1.0) {
            float xTexC = xC * ${origOutputDepth}.0 + d1;
            float wTexC = d1;

            // Read x(xR, xC, d1).
            vec2 xUV = (vec2(xTexC, xTexR) + halfCR) / xShapeCR;
            float xValue = texture2D(x, xUV).r;

            // Read w(wRPerm, wCPerm, d2, d1).
            vec2 wUV = (vec2(wTexC, wTexR) + halfCR) / wShapeCR;
            float wValue = texture2D(weights, wUV).r;

            dotProd += xValue * wValue;
          }
        }
      }
      ${biasOperation}
      gl_FragColor = vec4(dotProd, 0, 0, 0);
    }`;
}

export function getFragmentShaderDerBiasSource(
    dyShapeRCD: [number, number, number]) {
  const dyTexShapeRC = conv_util.computeTexShapeFrom3D(dyShapeRCD);
  const [yNumRows, yNumCols, outputDepth] = dyShapeRCD;

  return `
    precision highp float;
    uniform sampler2D dy;

    const vec2 halfCR = vec2(0.5, 0.5);
    const vec2 dyShapeCR = vec2(${dyTexShapeRC[1]}, ${dyTexShapeRC[0]});

    void main() {
      vec2 biasTexCR = floor(gl_FragCoord.xy);

      // The bias texture RC shape is [1, d2].
      float d2 = biasTexCR.x;

      float derBias = 0.0;
      for (float yR = 0.0; yR < ${yNumRows}.0; yR += 1.0) {
        float yTexR = yR;

        for (float yC = 0.0; yC < ${yNumCols}.0; yC += 1.0) {
          // Map from 3D (yR, yC, d2) to 2D (yTexR, yTexC).
          float yTexC = yC * ${outputDepth}.0 + d2;

          // Read dy(yR, yC, d2).
          vec2 dyUV = (vec2(yTexC, yTexR) + halfCR) / dyShapeCR;
          float dyValue = texture2D(dy, dyUV).r;

          derBias += dyValue;
        }
      }
      gl_FragColor = vec4(derBias, 0, 0, 0);
    }`;
}

export function derBias(
    gpgpu: GPGPUContext, program: WebGLProgram, dyTex: WebGLTexture,
    result: WebGLTexture, resultTexShapeRC: [number, number]) {
  gpgpu.setOutputMatrixTexture(
      result, resultTexShapeRC[0], resultTexShapeRC[1]);
  gpgpu.setProgram(program);
  gpgpu.setInputMatrixTexture(dyTex, 'dy', 0);
  gpgpu.executeProgram();
}

export function derWeights(
    gpgpu: GPGPUContext, program: WebGLProgram, xTex: WebGLTexture,
    dyTex: WebGLTexture, result: WebGLTexture,
    resultTexShapeRC: [number, number]) {
  gpgpu.setOutputMatrixTexture(
      result, resultTexShapeRC[0], resultTexShapeRC[1]);
  gpgpu.setProgram(program);
  gpgpu.setInputMatrixTexture(xTex, 'x', 0);
  gpgpu.setInputMatrixTexture(dyTex, 'dy', 1);
  gpgpu.executeProgram();
}

export function convTranspose(
    gpgpu: GPGPUContext, program: WebGLProgram, xTex: WebGLTexture,
    weightsTex: WebGLTexture, biasesTex: WebGLTexture|null,
    resultTex: WebGLTexture, resultTexShapeRC: [number, number]) {
  gpgpu.setOutputMatrixTexture(
      resultTex, resultTexShapeRC[0], resultTexShapeRC[1]);
  gpgpu.setProgram(program);
  gpgpu.setInputMatrixTexture(xTex, 'x', 0);
  gpgpu.setInputMatrixTexture(weightsTex, 'weights', 1);
  if (biasesTex != null) {
    gpgpu.setInputMatrixTexture(biasesTex, 'biases', 2);
  }
  gpgpu.executeProgram();
}
