/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

interface Dimensions {
  x?: number;
  y?: number;
  width: number;
  height: number;
  depth: 3|4;
}

// Shared cached frameBuffer object from external context
let fbo: WebGLFramebuffer;
let resizeProgram: WebGLProgram;
let resizeTexture: WebGLTexture;

let passThroughProgram: WebGLProgram;

export function downloadTextureData(
    gl: WebGL2RenderingContext, texture: WebGLTexture,
    dims: Dimensions): Uint8Array {
  const {x, y, width, height, depth} = dims;
  const pixels = new Uint8Array(width * height * depth);

  if (fbo == null) {
    fbo = createFrameBuffer(gl);
  }

  const LEVEL = 0;
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, LEVEL);

  const format = depth === 3 ? gl.RGB : gl.RGBA;
  gl.readPixels(x, y, width, height, format, gl.UNSIGNED_BYTE, pixels);

  // Unbind framebuffer
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return pixels;
}

export function uploadTextureData(
    imageData: Uint8Array, gl: WebGL2RenderingContext, dims: Dimensions,
    texture?: WebGLTexture): WebGLTexture {
  const targetTextureWidth = dims.width;
  const targetTextureHeight = dims.height;

  const targetTexture = texture || gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, targetTexture);

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  const level = 0;
  const format = dims.depth === 3 ? gl.RGB : gl.RGBA;
  const internalFormat = format;
  const border = 0;
  const type = gl.UNSIGNED_BYTE;

  gl.texImage2D(
      gl.TEXTURE_2D, level, internalFormat, targetTextureWidth,
      targetTextureHeight, border, format, type, imageData);

  return targetTexture;
}

export function runResizeProgram(
    gl: WebGL2RenderingContext, inputTexture: WebGLTexture,
    inputDims: Dimensions, outputDims: Dimensions) {
  if (resizeProgram == null) {
    resizeProgram = getResizeProgram(gl);
  }
  gl.useProgram(resizeProgram);

  //
  // Set up geometry
  //
  const positionAttrib = gl.getAttribLocation(resizeProgram, 'position');
  gl.enableVertexAttribArray(positionAttrib);
  gl.vertexAttribPointer(positionAttrib, 2, gl.FLOAT, false, 0, 0);

  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  const verts = new Float32Array([-2, 0, 0, -2, 2, 2]);
  gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);

  //
  // Set up input texutre
  //
  gl.uniform1i(gl.getUniformLocation(resizeProgram, 'inputTexture'), 1);
  gl.activeTexture(gl.TEXTURE0 + 1);
  gl.bindTexture(gl.TEXTURE_2D, inputTexture);

  //
  // Set up output texture.
  //
  if (resizeTexture == null) {
    resizeTexture = gl.createTexture();
  }
  const targetTexture = resizeTexture;
  const targetTextureWidth = outputDims.width;
  const targetTextureHeight = outputDims.height;

  gl.activeTexture(gl.TEXTURE0 + 2);
  gl.bindTexture(gl.TEXTURE_2D, targetTexture);

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  const level = 0;
  const internalFormat = gl.RGBA;
  const border = 0;
  const format = gl.RGBA;
  const type = gl.UNSIGNED_BYTE;
  gl.texImage2D(
      gl.TEXTURE_2D, level, internalFormat, targetTextureWidth,
      targetTextureHeight, border, format, type, null);

  // gl.bindTexture(gl.TEXTURE_2D, null);

  //
  // Render to output texture
  //
  if (fbo == null) {
    fbo = createFrameBuffer(gl);
  }

  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, targetTexture,
      level);

  const oldViewPort = gl.getParameter(gl.VIEWPORT);
  gl.viewport(0, 0, targetTextureWidth, targetTextureHeight);
  gl.clearColor(1, 0, 1, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  const NUM_VERTS = 6;
  gl.drawArrays(gl.TRIANGLES, 0, NUM_VERTS / 2);

  // Restore previous state
  gl.viewport(oldViewPort[0], oldViewPort[1], oldViewPort[2], oldViewPort[3]);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

  return targetTexture;
}

function createFrameBuffer(gl: WebGL2RenderingContext): WebGLFramebuffer {
  const fb = gl.createFramebuffer();
  const fboComplete = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (fboComplete !== gl.FRAMEBUFFER_COMPLETE) {
    console.log('initFrameBuffer: FRAMBUFFER IS NOT COMPLETE', fboComplete);
    switch (fboComplete) {
      case gl.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
        console.log('gl.FRAMEBUFFER_INCOMPLETE_ATTACHMENT');
        break;
      case gl.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
        console.log('gl.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT');
        break;
      case gl.FRAMEBUFFER_INCOMPLETE_DIMENSIONS:
        console.log('gl.FRAMEBUFFER_INCOMPLETE_DIMENSIONS');
        break;
      case gl.FRAMEBUFFER_UNSUPPORTED:
        console.log('gl.FRAMEBUFFER_UNSUPPORTED');
        break;
      case gl.FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
        console.log('gl.FRAMEBUFFER_INCOMPLETE_MULTISAMPLE');
        break;
      case gl.RENDERBUFFER_SAMPLES:
        console.log('gl.RENDERBUFFER_SAMPLES');
        break;
      default:
        console.log('unknown fbo complete status');
    }
  }
  return fb;
}

function getResizeProgram(gl: WebGL2RenderingContext): WebGLProgram {
  const vertShaderSource = `#version 300 es
precision highp float;
in vec2 position;
out vec2 uv;
void main() {
  uv = position;
  gl_Position = vec4(1.0 - 2.0 * position, 0, 1);
}`;

  const fragShaderSource2 = `#version 300 es
precision highp float;
uniform sampler2D inputTexture;
in vec2 uv;
out vec4 fragColor;
void main() {
  vec4 texSample = texture(inputTexture, uv);
  fragColor = vec4(128,texSample.g,texSample.b,255);
}`;
  //
  // Set up program
  //
  const vertShader = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vertShader, vertShaderSource);
  gl.compileShader(vertShader);

  const fragShader = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fragShader, fragShaderSource2);
  gl.compileShader(fragShader);

  const program = gl.createProgram();
  gl.attachShader(program, vertShader);
  gl.attachShader(program, fragShader);
  gl.linkProgram(program);
  gl.validateProgram(program);

  return program;
}


export function getPassthroughProgram(gl: WebGL2RenderingContext):
    WebGLProgram {
  if (passThroughProgram == null) {
    const vertShaderSource = `#version 300 es
precision highp float;
in vec2 position;
out vec2 uv;
void main() {
  uv = position;
  gl_Position = vec4(1.0 - 2.0 * position, 0, 1);
}`;

    const fragShaderSource2 = `#version 300 es
precision highp float;
uniform sampler2D inputTexture;
in vec2 uv;
out vec4 fragColor;
void main() {
  vec4 texSample = texture(inputTexture, uv);
  fragColor = vec4(128,texSample.g,texSample.b,255);
}`;
    //
    // Set up program
    //
    const vertShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertShader, vertShaderSource);
    gl.compileShader(vertShader);

    const fragShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragShader, fragShaderSource2);
    gl.compileShader(fragShader);

    const program = gl.createProgram();
    gl.attachShader(program, vertShader);
    gl.attachShader(program, fragShader);
    gl.linkProgram(program);
    gl.validateProgram(program);

    passThroughProgram = program;
  }
  return passThroughProgram;
}
