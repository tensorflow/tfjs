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

import {Array2D, GPGPUContext, webgl_util} from 'deeplearn';

export function createInputAtlas(
    imageSize: number, inputNumDimensions: number, numLatentVariables: number) {
  const coords = new Float32Array(
      imageSize * imageSize * (inputNumDimensions + numLatentVariables));
  let dst = 0;
  for (let d = 0; d < inputNumDimensions + numLatentVariables; d++) {
    for (let i = 0; i < imageSize * imageSize; i++) {
      const x = i % imageSize;
      const y = Math.floor(i / imageSize);
      const coord = imagePixelToNormalizedCoord(
          x, y, imageSize, imageSize, numLatentVariables);
      coords[dst++] = coord[d];
    }
  }

  return Array2D.new(
      [inputNumDimensions + numLatentVariables, imageSize * imageSize], coords);
}

export function getAddLatentVariablesShader(
    gpgpu: GPGPUContext, inputNumDimensions: number): WebGLProgram {
  const fragmentShaderSource = `
    precision highp float;
    uniform sampler2D source;
    varying vec2 resultUV;

    uniform vec2 z;

    const vec2 halfCR = vec2(0.5, 0.5);

    void main() {
      vec2 outputCR = floor(gl_FragCoord.xy);
      if (outputCR[1] == ${inputNumDimensions}.0) {
        gl_FragColor = vec4(z[0], 0, 0, 0);
      } else if (outputCR[1] > ${inputNumDimensions}.0) {
        gl_FragColor = vec4(z[1], 0, 0, 0);
      } else {
        gl_FragColor = texture2D(source, resultUV);
      }
    }`;
  return gpgpu.createProgram(fragmentShaderSource);
}

export function addLatentVariables(
    gpgpu: GPGPUContext, addZShader: WebGLProgram, sourceTex: WebGLTexture,
    resultTex: WebGLTexture, shapeRowCol: [number, number], z1: number,
    z2: number) {
  gpgpu.setOutputMatrixTexture(resultTex, shapeRowCol[0], shapeRowCol[1]);
  gpgpu.setProgram(addZShader);

  const sourceSamplerLocation = webgl_util.getProgramUniformLocationOrThrow(
      gpgpu.gl, addZShader, 'source');
  gpgpu.setInputMatrixTexture(sourceTex, sourceSamplerLocation, 0);

  const zLoc = gpgpu.getUniformLocation(addZShader, 'z');
  gpgpu.gl.uniform2f(zLoc, z1, z2);
  gpgpu.executeProgram();
}

export function getRenderShader(
    gpgpu: GPGPUContext, imageSize: number): WebGLProgram {
  const fragmentShaderSource = `
    precision highp float;
    uniform sampler2D source;
    varying vec2 resultUV;

    uniform int colorMode;
    uniform float outputNumDimensions;

    const float destinationSize = ${imageSize}.0;

    const mat3 yuv2rgb = mat3(
          1,       1,     1,
          0, -.34413, 1.772,
      1.402, -.71414,     0);

    vec3 hsv2rgb(vec3 c) {
      vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
      vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
      return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
    }

    void main() {
      vec2 outputCR = floor(gl_FragCoord.xy);
      float inputC = outputCR.y * destinationSize + outputCR.x;
      float u = (inputC + 0.5) / ${imageSize * imageSize}.0;

      vec4 inputR = vec4(0.0, 1.0, 2.0, 3.0);
      vec4 v = (inputR + 0.5) / outputNumDimensions;

      vec4 values = vec4(
        texture2D(source, vec2(u, v[0])).r,
        texture2D(source, vec2(u, v[1])).r,
        texture2D(source, vec2(u, v[2])).r,
        texture2D(source, vec2(u, v[3])).r);

      if (colorMode == 0) {
        // RGB
        gl_FragColor = vec4(values.rgb, 1.0);
      } else if (colorMode == 1) {
        // RGBA
        gl_FragColor = values;
      } else if (colorMode == 2) {
        // HSV
        vec3 rgb = hsv2rgb(values.rgb);
        gl_FragColor = vec4(rgb, 1.0);
      } else if (colorMode == 3) {
        // HSVA
        vec3 rgb = hsv2rgb(values.rgb);
        gl_FragColor = vec4(rgb, values[3]);
      } else if (colorMode == 4 || colorMode == 5) {
        // YUV
        values[0] = clamp(values[0], 0.2, 0.8);
        values[1] = values[1] - 0.5;
        values[2] = values[2] - 0.5;
        vec3 rgb = yuv2rgb * values.rgb;
        if (colorMode == 4) {
          // YUV
          gl_FragColor = vec4(rgb, 1.0);
        } else if (colorMode == 5) {
          // YUVA
          gl_FragColor = vec4(rgb, values.a);
        }
      } else if (colorMode == 6) {
        gl_FragColor = vec4(values[0], values[0], values[0], 1.0);
      }
    }`;

  return gpgpu.createProgram(fragmentShaderSource);
}

export function render(
    gpgpu: GPGPUContext, renderShader: WebGLProgram, sourceTex: WebGLTexture,
    outputNumDimensions: number, colorMode: number) {
  webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
  gpgpu.setProgram(renderShader);

  const sourceSamplerLocation = webgl_util.getProgramUniformLocationOrThrow(
      gpgpu.gl, renderShader, 'source');
  gpgpu.setInputMatrixTexture(sourceTex, sourceSamplerLocation, 0);
  const colorModeLoc = gpgpu.getUniformLocation(renderShader, 'colorMode');
  gpgpu.gl.uniform1i(colorModeLoc, colorMode);
  const outputNumDimensionsLoc =
      gpgpu.getUniformLocation(renderShader, 'outputNumDimensions');
  gpgpu.gl.uniform1f(outputNumDimensionsLoc, outputNumDimensions);
  gpgpu.executeProgram();
}

// Normalizes x, y to -.5 <=> +.5, adds a radius term, and pads zeros with the
// number of z parameters that will get added by the add z shader.
export function imagePixelToNormalizedCoord(
    x: number, y: number, imageWidth: number, imageHeight: number,
    zSize: number): number[] {
  const halfWidth = imageWidth * 0.5;
  const halfHeight = imageHeight * 0.5;
  const normX = (x - halfWidth) / imageWidth;
  const normY = (y - halfHeight) / imageHeight;

  const r = Math.sqrt(normX * normX + normY * normY);

  const result = [normX, normY, r];

  // Pad with zeros the number of latent terms, these get added on the GPU as
  // uniforms.
  for (let i = 0; i < zSize; i++) {
    result.push(0);
  }
  return result;
}
