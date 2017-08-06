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
import {IS_NAN_SHADER_FUNC} from './webgl_util';

export function getFragmentShaderPoolCommonSource(
    xShapeRCD: [number, number, number], fSize: number, stride: number,
    pad: number, poolType: 'max'|'min'|'avg', computePositions: boolean) {
  if (poolType === 'avg' && computePositions) {
    throw new Error('Cannot compute positions for average pool.');
  }

  const depth = xShapeRCD[2];

  const xTexShapeRC = conv_util.computeTexShapeFrom3D(xShapeRCD);

  let returnValue = 'minMaxValue';
  if (computePositions) {
    returnValue = 'minMaxPosition';
  } else if (poolType === 'avg') {
    returnValue = 'avgValue';
  }

  return `
    precision highp float;
    uniform sampler2D x;
    varying vec2 resultUV;

    const vec2 halfCR = vec2(0.5, 0.5);
    const vec2 xShapeCR = vec2(${xTexShapeRC[1]}, ${xTexShapeRC[0]});

    ${IS_NAN_SHADER_FUNC}

    void main() {
      vec2 yTexCR = floor(gl_FragCoord.xy);

      // Map from 2D (yTexR, yTexC) to 3D (yR, yC, d2).
      float yR = yTexCR.y;
      float yC = floor(yTexCR.x / ${depth}.0);
      float d = mod(yTexCR.x, ${depth}.0);

      vec2 xRCCorner = vec2(yR, yC) * vec2(${stride}, ${stride}) -
          vec2(${pad}.0, ${pad}.0);
      float xRCorner = xRCCorner.x;
      float xCCorner = xRCCorner.y;

      // max/min x(?, ?, d) to get y(yR, yC, d).
      // ? = to be determined
      float minMaxValue = 0.0;
      float minMaxValueFound = 0.0;
      float minMaxPosition = 0.0;
      float avgValue = 0.0;

      for (int wR = 0; wR < ${fSize}; wR++) {
        float wR_float = float(wR);
        float xR = xRCorner + wR_float;
        float xTexR = xR;

        for (int wC = 0; wC < ${fSize}; wC++) {
          float wC_float = float(wC);
          float xC = xCCorner + wC_float;
          float xTexC = xC * ${depth}.0 + d;

          vec2 texCR = vec2(xTexC, xTexR);

          // Check if the requested UV is invalid.
          vec2 uv = (texCR + halfCR) / xShapeCR;
          bool lessThanZero = any(lessThan(uv, vec2(0, 0)));
          bool greaterThanOne = any(greaterThan(uv, vec2(1, 1)));
          bool outside = lessThanZero || greaterThanOne;
          if (outside) {
            continue;
          }

          float value = texture2D(x, uv).r;
          if (isNaN(value)) {
            gl_FragColor = vec4(value, 0, 0, 0);
            return;
          }
          if (${poolType === 'avg'}) {
            avgValue += value / ${fSize * fSize}.0;
          } else {
            // If a min / max value has already been found, use it. If not, use
            // the current value.
            float currentMinMaxValue = mix(
                value, minMaxValue, minMaxValueFound);
            if (value ${poolType === 'min' ? '<=' : '>='} currentMinMaxValue) {
              minMaxValue = value;
              minMaxValueFound = 1.0;
              if (${computePositions}) {
                minMaxPosition = wR_float * ${fSize}.0 + wC_float;
              }
            }
          }
        }
      }
      gl_FragColor = vec4(${returnValue}, 0, 0, 0);
    }`;
}

export function poolCommon(
    gpgpu: GPGPUContext, program: WebGLProgram, x: WebGLTexture,
    result: WebGLTexture, resultShapeRowCol: [number, number]) {
  gpgpu.setOutputMatrixTexture(
      result, resultShapeRowCol[0], resultShapeRowCol[1]);
  gpgpu.setProgram(program);
  gpgpu.setInputMatrixTexture(x, 'x', 0);
  gpgpu.executeProgram();
}
