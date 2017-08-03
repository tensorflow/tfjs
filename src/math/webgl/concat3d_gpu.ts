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
    x1ShapeRCD: [number, number, number], x2ShapeRCD: [number, number, number],
    resultShapeRCD: [number, number, number], axis: number): string {
  const x1TexShapeRC = conv_util.computeTexShapeFrom3D(x1ShapeRCD);
  const x2TexShapeRC = conv_util.computeTexShapeFrom3D(x2ShapeRCD);

  const yAxes = ['yR', 'yC', 'yD'];
  const concatAxis = yAxes[axis];

  return `
    precision highp float;
    uniform sampler2D x1;
    uniform sampler2D x2;

    const vec2 x1ShapeCR = vec2(${x1TexShapeRC[1]}, ${x1TexShapeRC[0]});
    const vec2 x2ShapeCR = vec2(${x2TexShapeRC[1]}.0, ${x2TexShapeRC[0]}.0);

    const vec2 halfCR = vec2(0.5, 0.5);

    void main() {
      vec2 yTexCR = floor(gl_FragCoord.xy);

      // Map from 2D (yTexR, yTexC) to 3D (yR, yC, yD).
      float yR = yTexCR.y;
      float yC = floor(yTexCR.x / ${resultShapeRCD[2]}.0);
      float yD = mod(yTexCR.x, ${resultShapeRCD[2]}.0);

      float value = 0.0;

      if (${concatAxis} < ${x1ShapeRCD[axis]}.0) {
        // Map yR, yC, yD back to x1 coordinates.
        vec2 x1CR = vec2(yC * ${x1ShapeRCD[2]}.0 + yD, yR);
        vec2 x1UV = (x1CR + halfCR) / x1ShapeCR;
        value = texture2D(x1, x1UV).r;
      } else {
        ${concatAxis} = ${concatAxis} - ${x1ShapeRCD[axis]}.0;

        // Map yR, yC, yD back to x2 coordinates.
        vec2 x2CR = vec2(yC * ${x2ShapeRCD[2]}.0 + yD, yR);
        vec2 x2UV = (x2CR + halfCR) / x2ShapeCR;
        value = texture2D(x2, x2UV).r;
      }

      gl_FragColor = vec4(value, 0.0, 0.0, 0.0);
    }`;
}

export function concat3D(
    gpgpu: GPGPUContext, program: WebGLProgram, x1: WebGLTexture,
    x2: WebGLTexture, result: WebGLTexture, resultShapeRC: [number, number]) {
  gpgpu.setOutputMatrixTexture(result, resultShapeRC[0], resultShapeRC[1]);
  gpgpu.setProgram(program);
  gpgpu.setInputMatrixTexture(x1, 'x1', 0);
  gpgpu.setInputMatrixTexture(x2, 'x2', 1);
  gpgpu.executeProgram();
}
