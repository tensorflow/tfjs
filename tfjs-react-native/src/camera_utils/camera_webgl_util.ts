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

import * as tf from '@tensorflow/tfjs-core';

import * as drawTextureProgramInfo from './draw_texture_program_info';
import {m4} from './matrix_utils';
import * as resizeProgramInfo from './resize_program_info';

interface Dimensions {
  x?: number;
  y?: number;
  width: number;
  height: number;
  depth: number;
}

// Shared cached frameBuffer object from external context
let fbo: WebGLFramebuffer;

// Internal target texture used for resizing camera texture input
let resizeTexture: WebGLTexture;
let resizeTextureDims: {width: number, height: number};

interface ProgramObjects {
  program: WebGLProgram;
  vao: WebGLVertexArrayObject;
  vertices: Float32Array;
}

// Cache for shader programs and associated vertex array buffers.
const programCache: Map<string, ProgramObjects> = new Map();

/**
 * Download data from an texture.
 * @param gl
 * @param texture
 * @param dims
 */
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
  gl.bindTexture(gl.TEXTURE_2D, texture);

  gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, LEVEL);

  const format = depth === 3 ? gl.RGB : gl.RGBA;
  tf.webgl.webgl_util.callAndCheck(gl, true, () => {
    gl.readPixels(x, y, width, height, format, gl.UNSIGNED_BYTE, pixels);
  });

  // Unbind framebuffer
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return pixels;
}

/**
 * Upload image data to a texture.
 *
 * @param imageData data to upload
 * @param gl gl context to use
 * @param dims image size
 * @param texture optional texture to upload data to. If none is passed a new
 *     texture will be returned
 */
export function uploadTextureData(
    imageData: Uint8Array, gl: WebGL2RenderingContext, dims: Dimensions,
    texture?: WebGLTexture): WebGLTexture {
  const targetTextureWidth = dims.width;
  const targetTextureHeight = dims.height;

  tf.util.assert(
      targetTextureWidth * targetTextureHeight * dims.depth ===
          imageData.length,
      () => 'uploadTextureData Error: imageData length must match w * h * d');

  const targetTexture = texture || gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, targetTexture);

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  const level = 0;
  const format = dims.depth === 3 ? gl.RGB : gl.RGBA;
  const internalFormat = format;
  const border = 0;
  const type = gl.UNSIGNED_BYTE;

  tf.webgl.webgl_util.callAndCheck(gl, true, () => {
    gl.texImage2D(
        gl.TEXTURE_2D, level, internalFormat, targetTextureWidth,
        targetTextureHeight, border, format, type, imageData);
  });

  gl.bindTexture(gl.TEXTURE_2D, null);
  return targetTexture;
}

/**
 * WIP Render a texture to the default framebuffer (i.e. screen)
 * @param gl WebGL context to use
 * @param texture texture to render
 * @param dims texture size
 */
export function drawTexture(
    gl: WebGL2RenderingContext, texture: WebGLTexture, dims: Dimensions) {
  const {program, vao, vertices} = drawTextureProgram(gl);
  gl.useProgram(program);
  gl.bindVertexArray(vao);

  // Set texture sampler uniform
  const TEXTURE_UNIT = 0;
  gl.uniform1i(gl.getUniformLocation(program, 'inputTexture'), TEXTURE_UNIT);
  gl.activeTexture(gl.TEXTURE0 + TEXTURE_UNIT);
  gl.bindTexture(gl.TEXTURE_2D, texture);

  let matrix = m4.orthographic(
      0, gl.drawingBufferWidth, gl.drawingBufferHeight, 0, -1, 1);
  matrix = m4.translate(matrix, 0, 0, 0);
  matrix = m4.scale(matrix, dims.width, dims.height, 1);

  // console.log('matrix', matrix);
  gl.uniformMatrix4fv(
      gl.getUniformLocation(program, 'u_matrix'), false, matrix);

  // Draw to screen
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.clearColor(1, 1, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.drawArrays(gl.TRIANGLES, 0, vertices.length / 2);

  gl.bindVertexArray(null);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.useProgram(null);
}

export function runResizeProgram(
    gl: WebGL2RenderingContext, inputTexture: WebGLTexture,
    inputDims: Dimensions, outputDims: Dimensions) {
  const {program, vao, vertices} = getResizeProgram(gl, inputDims, outputDims);
  gl.useProgram(program);
  // Set up geometry
  tf.webgl.webgl_util.callAndCheck(gl, true, () => {
    gl.bindVertexArray(vao);
  });

  //
  // Set up input texutre
  //
  gl.uniform1i(gl.getUniformLocation(program, 'inputTexture'), 1);
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

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  // Reallocate texture if target size has changed.
  if (resizeTextureDims == null ||
      resizeTextureDims.width !== targetTextureWidth ||
      resizeTextureDims.height !== targetTextureHeight) {
    const level = 0;
    const internalFormat = gl.RGBA;
    const border = 0;
    const format = gl.RGBA;
    const type = gl.UNSIGNED_BYTE;
    gl.texImage2D(
        gl.TEXTURE_2D, level, internalFormat, targetTextureWidth,
        targetTextureHeight, border, format, type, null);
    resizeTextureDims = {
      width: targetTextureWidth,
      height: targetTextureHeight
    };
  }

  //
  // Render to output texture
  //
  if (fbo == null) {
    fbo = createFrameBuffer(gl);
  }

  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, targetTexture, 0);

  gl.clearColor(1, 0, 1, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, vertices.length / 2);

  // Restore previous state
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.useProgram(null);
  gl.bindVertexArray(null);
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

function drawTextureProgram(gl: WebGL2RenderingContext): ProgramObjects {
  const cacheKey = `drawTexture`;
  if (!programCache.has(cacheKey)) {
    const vertSource = drawTextureProgramInfo.vertexShaderSource();
    const fragSource = drawTextureProgramInfo.fragmentShaderSource();

    const vertices = drawTextureProgramInfo.vertices();
    const texCoords = drawTextureProgramInfo.texCoords();

    const programObjects =
        createProgramObjects(gl, vertSource, fragSource, vertices, texCoords);

    programCache.set(cacheKey, programObjects);
  }
  return programCache.get(cacheKey);
}

function getResizeProgram(
    gl: WebGL2RenderingContext, sourceDims: Dimensions,
    targetDims: Dimensions): ProgramObjects {
  const cacheKey = `resize_${targetDims.depth}`;
  if (!programCache.has(cacheKey)) {
    const vertSource = resizeProgramInfo.vertexShaderSource();
    const fragSource =
        resizeProgramInfo.fragmentShaderSource(sourceDims, targetDims, false);
    const vertices = resizeProgramInfo.vertices();
    const texCoords = resizeProgramInfo.texCoords();
    const programObjects =
        createProgramObjects(gl, vertSource, fragSource, vertices, texCoords);

    programCache.set(cacheKey, programObjects);
  }
  return programCache.get(cacheKey);
}

function createProgramObjects(
    gl: WebGL2RenderingContext, vertexShaderSource: string,
    fragmentShaderSource: string, vertices: Float32Array,
    texCoords: Float32Array): ProgramObjects {
  const vertShader = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vertShader, vertexShaderSource);
  gl.compileShader(vertShader);
  if (gl.getShaderParameter(vertShader, gl.COMPILE_STATUS) === false) {
    const errorString =
        JSON.parse(JSON.stringify(gl.getShaderInfoLog(vertShader)));
    console.log('Error compiling vertex shader\n', errorString);
  }

  const fragShader = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fragShader, fragmentShaderSource);
  gl.compileShader(fragShader);
  if (gl.getShaderParameter(fragShader, gl.COMPILE_STATUS) === false) {
    const errorString =
        JSON.parse(JSON.stringify(gl.getShaderInfoLog(fragShader)));
    console.log('Error compiling fragment shader\n', errorString);
  }

  const program = gl.createProgram();
  gl.attachShader(program, vertShader);
  gl.attachShader(program, fragShader);
  gl.linkProgram(program);
  gl.validateProgram(program);
  if (gl.getProgramParameter(program, gl.VALIDATE_STATUS) === false) {
    console.log(gl.getProgramInfoLog(program));
    throw new Error('Shader program validation failed.');
  }

  // vao allows us to store geometry settings.
  const vao = gl.createVertexArray();

  gl.useProgram(program);
  gl.bindVertexArray(vao);

  // Set up geometry
  const positionAttrib = gl.getAttribLocation(program, 'position');
  if (positionAttrib === -1) {
    console.log('Could not get attribute location for "position"');
    console.log('vertex shader', gl.getShaderSource(vertShader));
    console.log('fragment shader', gl.getShaderSource(fragShader));
  }
  tf.webgl.webgl_util.callAndCheck(gl, true, () => {
    const vertsCoordsBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertsCoordsBuffer);
    gl.enableVertexAttribArray(positionAttrib);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    gl.vertexAttribPointer(positionAttrib, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    console.log('Done setting VERTS_COORDS info');
  });

  const texCoordsAttrib = gl.getAttribLocation(program, 'texCoords');
  if (texCoordsAttrib === -1) {
    console.log('Could not get attribute location for "texCoords"');
    console.log('vertex shader', gl.getShaderSource(vertShader));
    console.log('fragment shader', gl.getShaderSource(fragShader));
  } else {
    tf.webgl.webgl_util.callAndCheck(gl, true, () => {
      const texCoordsBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, texCoordsBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(texCoordsAttrib);
      gl.vertexAttribPointer(texCoordsAttrib, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);
      console.log('Done setting TEX_COORDS info');
    });
  }

  // Unbind
  gl.bindVertexArray(null);
  gl.useProgram(null);
  return {
    program,
    vao,
    vertices,
  };
}
