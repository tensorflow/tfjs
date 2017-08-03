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

export function getFragmentShaderMaxPoolBackprop(
    dyShapeRCD: [number, number, number], fSize: number, origStride: number,
    origPad: number) {
  const origInputDepth = dyShapeRCD[2];
  const pad = fSize - 1 - origPad;
  const [dyRows, dyCols, depth] = dyShapeRCD;

  const dyTexShapeRC = conv_util.computeTexShapeFrom3D(dyShapeRCD);

  return `
    precision highp float;
    uniform sampler2D dy;
    uniform sampler2D maxPos;

    const vec2 halfCR = vec2(0.5, 0.5);
    const vec2 dyShapeCR = vec2(${dyTexShapeRC[1]}, ${dyTexShapeRC[0]});

    void main() {
      vec2 dxTexCR = floor(gl_FragCoord.xy);

      // Map from 2D (dxTexR, dxTexC) to 3D (dxR, dxC, d).
      float dxR = dxTexCR.y;
      float dxC = floor(dxTexCR.x / ${origInputDepth}.0);
      float d = mod(dxTexCR.x, ${origInputDepth}.0);

      vec2 dyRCCorner = vec2(dxR, dxC) - vec2(${pad}.0, ${pad}.0);
      float dyRCorner = dyRCCorner.x;
      float dyCCorner = dyRCCorner.y;

      // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(yR, dxC, d).
      // ? = to be determined. : = across all values in that axis.
      float dotProd = 0.0;
      for (float wR = 0.0; wR < ${fSize}.0; wR += 1.0) {

        float dyR = (dyRCorner + wR) / ${origStride}.0;
        // TODO(nsthorat): Splice this with another version where you call
        // getMatrixValueOrZeroPad(). Here and below.
        if (dyR < 0.0 || dyR >= ${dyRows}.0 || fract(dyR) > 0.0) {
          continue;
        }

        float dyTexR = dyR;

        for (float wC = 0.0; wC < ${fSize}.0; wC += 1.0) {

          float dyC = (dyCCorner + wC) / ${origStride}.0;
          if (dyC < 0.0 || dyC >= ${dyCols}.0 || fract(dyC) > 0.0) {
            continue;
          }

          float dyTexC = dyC * ${depth}.0 + d;

          // Read dy(dyR, dyC, d).
          vec2 dyUV = (vec2(dyTexC, dyTexR) + halfCR) / dyShapeCR;
          float dyValue = texture2D(dy, dyUV).r;

          // Read maxPos(dyR, dyC, d).
          float maxPosValue =
              ${fSize * fSize - 1}.0 - texture2D(maxPos, dyUV).r;

          // Get the current value, check it against the value from the
          // position matrix.
          float curPosValue = wR * ${fSize}.0 + wC;
          float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);

          dotProd += dyValue * mask;
        }
      }
      gl_FragColor = vec4(dotProd, 0, 0, 0);
    }`;
}

export function maxPoolBackprop(
    gpgpu: GPGPUContext, program: WebGLProgram, dyTex: WebGLTexture,
    maxPositionsTex: WebGLTexture, resultTex: WebGLTexture,
    resultTexShapeRC: [number, number]) {
  gpgpu.setOutputMatrixTexture(
      resultTex, resultTexShapeRC[0], resultTexShapeRC[1]);
  gpgpu.setProgram(program);
  gpgpu.setInputMatrixTexture(dyTex, 'dy', 0);
  gpgpu.setInputMatrixTexture(maxPositionsTex, 'maxPos', 1);
  gpgpu.executeProgram();
}
