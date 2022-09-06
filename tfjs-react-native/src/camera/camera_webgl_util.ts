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

import {webgl_util} from '@tensorflow/tfjs-backend-webgl';
import * as tf from '@tensorflow/tfjs-core';

import * as drawTextureProgramInfo from './draw_texture_program_info';
import * as resizeBilinearProgramInfo from './resize_bilinear_program_info';
import * as resizeNNProgramInfo from './resize_nearest_neigbor_program_info';
import {Rotation} from './types';

interface Dimensions {
  width: number;
  height: number;
  depth: number;
}

// Shared cached frameBuffer object from external context
const fboCache = new WeakMap<WebGL2RenderingContext, WebGLFramebuffer>();

// Internal target texture used for resizing camera texture input
const resizeTextureCache = new WeakMap<WebGL2RenderingContext, WebGLTexture>();
const resizeTextureDimsCache =
    new WeakMap<WebGL2RenderingContext, {width: number, height: number}>();

interface ProgramObjects {
  program: WebGLProgram;
  vao: WebGLVertexArrayObject;
  vertices: Float32Array;
  uniformLocations: Map<string, WebGLUniformLocation>;
}

// Cache for shader programs and associated vertex array buffers.
const programCacheByContext:
    WeakMap<WebGL2RenderingContext, Map<string, ProgramObjects>> =
        new WeakMap();

/**
 * Download data from an texture.
 *
 * @param gl
 * @param texture
 * @param dims
 */
export function downloadTextureData(
    gl: WebGL2RenderingContext, texture: WebGLTexture,
    dims: Dimensions): Uint8Array {
  const {width, height, depth} = dims;
  const pixels = new Uint8Array(width * height * depth);

  if (!fboCache.has(gl)) {
    fboCache.set(gl, createFrameBuffer(gl));
  }
  const fbo = fboCache.get(gl);

  webgl_util.callAndCheck(gl, () => {
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  });

  webgl_util.callAndCheck(gl, () => {
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
  });

  webgl_util.callAndCheck(gl, () => {
    const level = 0;
    gl.framebufferTexture2D(
        gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, level);
  });

  webgl_util.callAndCheck(gl, () => {
    const format = depth === 3 ? gl.RGB : gl.RGBA;
    const x = 0;
    const y = 0;
    gl.readPixels(x, y, width, height, format, gl.UNSIGNED_BYTE, pixels);
  });

  // Unbind framebuffer
  webgl_util.callAndCheck(gl, () => {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  });
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
  gl.activeTexture(gl.TEXTURE0);
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

  webgl_util.callAndCheck(gl, () => {
    gl.texImage2D(
        gl.TEXTURE_2D, level, internalFormat, targetTextureWidth,
        targetTextureHeight, border, format, type, imageData);
  });

  gl.bindTexture(gl.TEXTURE_2D, null);
  return targetTexture;
}

/**
 * Render a texture to the default framebuffer (i.e. screen)
 *
 * @param gl WebGL context to use
 * @param texture texture to render
 * @param dims texture size
 */
export function drawTexture(
    gl: WebGL2RenderingContext, texture: WebGLTexture,
    dims: {width: number, height: number}, flipHorizontal: boolean,
    rotation: Rotation) {
  const {program, vao, vertices, uniformLocations} =
      drawTextureProgram(gl, flipHorizontal, false, rotation);
  gl.useProgram(program);
  gl.bindVertexArray(vao);

  // Set texture sampler uniform
  const TEXTURE_UNIT = 0;
  gl.uniform1i(uniformLocations.get('inputTexture'), TEXTURE_UNIT);
  gl.activeTexture(gl.TEXTURE0 + TEXTURE_UNIT);
  gl.bindTexture(gl.TEXTURE_2D, texture);

  // Draw to screen
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, dims.width, dims.height);
  // gl.clearColor(1, 1, 0, 1);
  // gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.drawArrays(gl.TRIANGLES, 0, vertices.length / 2);

  gl.bindVertexArray(null);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.useProgram(null);
}

export function runResizeProgram(
    gl: WebGL2RenderingContext, inputTexture: WebGLTexture,
    inputDims: Dimensions, outputDims: Dimensions, alignCorners: boolean,
    useCustomShadersToResize: boolean,
    interpolation: 'nearest_neighbor'|'bilinear', rotation: Rotation) {
  const {program, vao, vertices, uniformLocations} = useCustomShadersToResize ?
      resizeProgram(gl, inputDims, outputDims, alignCorners, interpolation) :
      drawTextureProgram(gl, false, true, rotation);
  gl.useProgram(program);
  // Set up geometry
  webgl_util.callAndCheck(gl, () => {
    gl.bindVertexArray(vao);
  });

  //
  // Set up input texture
  //
  gl.uniform1i(uniformLocations.get('inputTexture'), 1);
  gl.activeTexture(gl.TEXTURE0 + 1);
  gl.bindTexture(gl.TEXTURE_2D, inputTexture);
  if (!useCustomShadersToResize) {
    const textureFilter =
        interpolation === 'nearest_neighbor' ? gl.NEAREST : gl.LINEAR;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, textureFilter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, textureFilter);
  }

  //
  // Set up output texture.
  //
  if (!resizeTextureCache.has(gl)) {
    resizeTextureCache.set(gl, gl.createTexture());
  }
  const resizeTexture = resizeTextureCache.get(gl);

  const targetTexture = resizeTexture;
  const targetTextureWidth = outputDims.width;
  const targetTextureHeight = outputDims.height;

  gl.activeTexture(gl.TEXTURE0 + 2);
  gl.bindTexture(gl.TEXTURE_2D, targetTexture);

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  // Reallocate texture storage if target size has changed.
  if (!resizeTextureDimsCache.has(gl)) {
    resizeTextureDimsCache.set(gl, {width: -1, height: -1});
  }
  const resizeTextureDims = resizeTextureDimsCache.get(gl);

  if (resizeTextureDims == null ||
      resizeTextureDims.width !== targetTextureWidth ||
      resizeTextureDims.height !== targetTextureHeight) {
    const level = 0;
    const format = outputDims.depth === 3 ? gl.RGB : gl.RGBA;
    const internalFormat = format;
    const border = 0;
    const type = gl.UNSIGNED_BYTE;

    webgl_util.callAndCheck(gl, () => {
      gl.texImage2D(
          gl.TEXTURE_2D, level, internalFormat, targetTextureWidth,
          targetTextureHeight, border, format, type, null);
    });

    resizeTextureDimsCache.set(
        gl, {width: targetTextureWidth, height: targetTextureHeight});
  }

  //
  // Render to output texture
  //
  if (!fboCache.has(gl)) {
    fboCache.set(gl, createFrameBuffer(gl));
  }
  const fbo = fboCache.get(gl);

  gl.viewport(0, 0, targetTextureWidth, targetTextureHeight);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, targetTexture, 0);

  const fboComplete = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (fboComplete !== gl.FRAMEBUFFER_COMPLETE) {
    switch (fboComplete) {
      case gl.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
        throw new Error(
            'createFrameBuffer: gl.FRAMEBUFFER_INCOMPLETE_ATTACHMENT');

      case gl.FRAMEBUFFER_UNSUPPORTED:
        throw new Error('createFrameBuffer: gl.FRAMEBUFFER_UNSUPPORTED');

      case gl.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
        throw new Error(
            'createFrameBuffer: gl.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT');

      case gl.FRAMEBUFFER_INCOMPLETE_DIMENSIONS:
        throw new Error(
            'createFrameBuffer: gl.FRAMEBUFFER_INCOMPLETE_DIMENSIONS');

      case gl.FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
        throw new Error(
            'createFrameBuffer: gl.FRAMEBUFFER_INCOMPLETE_MULTISAMPLE');
      default:
        throw new Error(
            'createFrameBuffer Error: Other or unknown fbo complete status: ' +
            `${fboComplete}`);
    }
  }

  gl.drawArrays(gl.TRIANGLES, 0, vertices.length / 2);

  // Restore previous state
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.useProgram(null);
  gl.bindVertexArray(null);
  return targetTexture;
}

function createFrameBuffer(gl: WebGL2RenderingContext): WebGLFramebuffer {
  const fb = gl.createFramebuffer();
  if (fb == null) {
    throw new Error('Could not create framebuffer');
  }
  return fb;
}

export function drawTextureProgram(
    gl: WebGL2RenderingContext, flipHorizontal: boolean, flipVertical: boolean,
    rotation: Rotation): ProgramObjects {
  if (!programCacheByContext.has(gl)) {
    programCacheByContext.set(gl, new Map());
  }
  const programCache = programCacheByContext.get(gl);

  const cacheKey = `drawTexture_${flipHorizontal}_${flipVertical}_${rotation}`;
  if (!programCache.has(cacheKey)) {
    const vertSource = drawTextureProgramInfo.vertexShaderSource(
        flipHorizontal, flipVertical, rotation);
    const fragSource = drawTextureProgramInfo.fragmentShaderSource();

    const vertices = drawTextureProgramInfo.vertices();
    const texCoords = drawTextureProgramInfo.texCoords();

    const programObjects =
        createProgramObjects(gl, vertSource, fragSource, vertices, texCoords);

    programCache.set(cacheKey, programObjects);
  }
  return programCache.get(cacheKey);
}

function resizeProgram(
    gl: WebGL2RenderingContext, sourceDims: Dimensions, targetDims: Dimensions,
    alignCorners: boolean,
    interpolation: 'nearest_neighbor'|'bilinear'): ProgramObjects {
  if (!programCacheByContext.has(gl)) {
    programCacheByContext.set(gl, new Map());
  }
  const programCache = programCacheByContext.get(gl);

  const cacheKey = `resize_${sourceDims.width}_${sourceDims.height}_${
      sourceDims.depth}_${targetDims.width}_${targetDims.height}_${
      targetDims.depth}_${alignCorners}_${interpolation}`;

  if (!programCache.has(cacheKey)) {
    const vertSource = resizeNNProgramInfo.vertexShaderSource();
    let fragSource: string;
    if (interpolation === 'nearest_neighbor') {
      fragSource = resizeNNProgramInfo.fragmentShaderSource(
          sourceDims, targetDims, alignCorners);
    } else {
      fragSource = resizeBilinearProgramInfo.fragmentShaderSource(
          sourceDims, targetDims, alignCorners);
    }

    const vertices = resizeNNProgramInfo.vertices();
    const texCoords = resizeNNProgramInfo.texCoords();
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

  const fragShader = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fragShader, fragmentShaderSource);
  gl.compileShader(fragShader);

  const program = gl.createProgram();
  gl.attachShader(program, vertShader);
  gl.attachShader(program, fragShader);
  gl.linkProgram(program);
  gl.validateProgram(program);

  // Use a vertex array objects to record geometry info
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);

  // Set up geometry
  webgl_util.callAndCheck(gl, () => {
    const positionAttrib = gl.getAttribLocation(program, 'position');
    const vertsCoordsBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertsCoordsBuffer);
    gl.enableVertexAttribArray(positionAttrib);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    gl.vertexAttribPointer(positionAttrib, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  });

  webgl_util.callAndCheck(gl, () => {
    const texCoordsAttrib = gl.getAttribLocation(program, 'texCoords');
    const texCoordsBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordsBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(texCoordsAttrib);
    gl.vertexAttribPointer(texCoordsAttrib, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  });

  const uniformLocations = new Map<string, WebGLUniformLocation>();
  webgl_util.callAndCheck(gl, () => {
    const inputTextureLoc = gl.getUniformLocation(program, 'inputTexture');
    uniformLocations.set('inputTexture', inputTextureLoc);
  });

  // Unbind
  gl.bindVertexArray(null);
  return {
    program,
    vao,
    vertices,
    uniformLocations,
  };
}
