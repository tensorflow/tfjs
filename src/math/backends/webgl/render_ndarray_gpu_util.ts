/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {GPGPUContext} from './gpgpu_context';

import * as webgl_util from './webgl_util';

export function getRenderRGBShader(
    gpgpu: GPGPUContext, destinationWidth: number): WebGLProgram {
  const fragmentShaderSource = `
    precision highp float;
    uniform sampler2D source;
    varying vec2 resultUV;

    const float destinationWidth = ${destinationWidth}.0;
    const float a = 1.0;

    void main() {
      float xr = floor(resultUV.s * destinationWidth) * 3.0;
      vec3 x = xr + vec3(0, 1, 2);

      float sourceWidth = destinationWidth * 3.0;
      vec3 u = (x + 0.5) / sourceWidth;
      float v = 1.0 - resultUV.t;

      float r = texture2D(source, vec2(u[0], v)).r;
      float g = texture2D(source, vec2(u[1], v)).r;
      float b = texture2D(source, vec2(u[2], v)).r;

      gl_FragColor = vec4(r, g, b, a);
    }`;

  return gpgpu.createProgram(fragmentShaderSource);
}

export function renderToCanvas(
    gpgpu: GPGPUContext, renderShader: WebGLProgram, sourceTex: WebGLTexture) {
  webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
  renderToFramebuffer(gpgpu, renderShader, sourceTex);
}

export function renderToFramebuffer(
    gpgpu: GPGPUContext, renderShader: WebGLProgram, sourceTex: WebGLTexture) {
  gpgpu.setProgram(renderShader);

  const sourceSamplerLocation = webgl_util.getProgramUniformLocationOrThrow(
      gpgpu.gl, renderShader, 'source');
  gpgpu.setInputMatrixTexture(sourceTex, sourceSamplerLocation, 0);
  gpgpu.executeProgram();
}
