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
    sourceShapeRowCol: [number, number], sourceSizeRowCol: [number, number],
    destSizeRowCol: [number, number]): string {
  return `
    precision highp float;
    uniform sampler2D source;
    uniform vec2 sourceStartCR;
    uniform vec2 destStartCR;

    const vec2 sourceShapeCR =
      vec2(${sourceShapeRowCol[1]}, ${sourceShapeRowCol[0]});
    const vec2 sourceSizeCR =
      vec2(${sourceSizeRowCol[1]}, ${sourceSizeRowCol[0]});
    const vec2 destSizeCR =
      vec2(${destSizeRowCol[1]}, ${destSizeRowCol[0]});

    void main() {
      vec2 destOffsetCR = floor(gl_FragCoord.xy) - destStartCR;
      float destOffsetFlat = (destOffsetCR.y * destSizeCR.x) + destOffsetCR.x;
      vec2 sourceOffsetCR = vec2(mod(destOffsetFlat, sourceSizeCR.x),
        floor(destOffsetFlat / sourceSizeCR.x));
      vec2 sourceCR = sourceStartCR + sourceOffsetCR;
      vec2 sourceUV = (sourceCR + vec2(0.5, 0.5)) / sourceShapeCR;
      gl_FragColor = texture2D(source, sourceUV);
    }`;
}

export function copy(
    gpgpu: GPGPUContext, program: WebGLProgram, source: WebGLTexture,
    sourceShapeRowCol: [number, number], sourceStartRowCol: [number, number],
    sourceSizeRowCol: [number, number], dest: WebGLTexture,
    destShapeRowCol: [number, number], destStartRowCol: [number, number],
    destSizeRowCol: [number, number]) {
  gpgpu.setOutputMatrixTexture(dest, destShapeRowCol[0], destShapeRowCol[1]);
  gpgpu.setOutputMatrixWriteRegion(
      destStartRowCol[0], destSizeRowCol[0], destStartRowCol[1],
      destSizeRowCol[1]);
  gpgpu.setProgram(program);
  gpgpu.setInputMatrixTexture(source, 'source', 0);
  const sourceStartCRLoc = gpgpu.getUniformLocation('sourceStartCR');
  gpgpu.gl.uniform2f(
      sourceStartCRLoc, sourceStartRowCol[1], sourceStartRowCol[0]);
  const destStartCRLoc = gpgpu.getUniformLocation('destStartCR');
  gpgpu.gl.uniform2f(destStartCRLoc, destStartRowCol[1], destStartRowCol[0]);
  gpgpu.executeProgram();
}
