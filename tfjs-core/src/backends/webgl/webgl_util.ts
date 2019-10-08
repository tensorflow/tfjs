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

import {env} from '../../environment';

import * as util from '../../util';

import {getWebGLContext} from './canvas_util';
import {getTextureConfig} from './tex_util';

export function callAndCheck<T>(
    gl: WebGLRenderingContext, debugMode: boolean, func: () => T): T {
  const returnValue = func();
  if (debugMode) {
    checkWebGLError(gl);
  }
  return returnValue;
}

function checkWebGLError(gl: WebGLRenderingContext) {
  const error = gl.getError();
  if (error !== gl.NO_ERROR) {
    throw new Error('WebGL Error: ' + getWebGLErrorMessage(gl, error));
  }
}

// https://en.wikipedia.org/wiki/Half-precision_floating-point_format
const MIN_FLOAT16 = 5.96e-8;
const MAX_FLOAT16 = 65504;

export function canBeRepresented(num: number): boolean {
  if (env().getBool('WEBGL_RENDER_FLOAT32_ENABLED') || num === 0 ||
      (MIN_FLOAT16 < Math.abs(num) && Math.abs(num) < MAX_FLOAT16)) {
    return true;
  }
  return false;
}

export function getWebGLErrorMessage(
    gl: WebGLRenderingContext, status: number): string {
  switch (status) {
    case gl.NO_ERROR:
      return 'NO_ERROR';
    case gl.INVALID_ENUM:
      return 'INVALID_ENUM';
    case gl.INVALID_VALUE:
      return 'INVALID_VALUE';
    case gl.INVALID_OPERATION:
      return 'INVALID_OPERATION';
    case gl.INVALID_FRAMEBUFFER_OPERATION:
      return 'INVALID_FRAMEBUFFER_OPERATION';
    case gl.OUT_OF_MEMORY:
      return 'OUT_OF_MEMORY';
    case gl.CONTEXT_LOST_WEBGL:
      return 'CONTEXT_LOST_WEBGL';
    default:
      return `Unknown error code ${status}`;
  }
}

export function getExtensionOrThrow(
    gl: WebGLRenderingContext, debug: boolean, extensionName: string): {} {
  return throwIfNull<{}>(
      gl, debug, () => gl.getExtension(extensionName),
      'Extension "' + extensionName + '" not supported on this browser.');
}

export function createVertexShader(
    gl: WebGLRenderingContext, debug: boolean,
    vertexShaderSource: string): WebGLShader {
  const vertexShader: WebGLShader = throwIfNull<WebGLShader>(
      gl, debug, () => gl.createShader(gl.VERTEX_SHADER),
      'Unable to create vertex WebGLShader.');
  callAndCheck(
      gl, debug, () => gl.shaderSource(vertexShader, vertexShaderSource));
  callAndCheck(gl, debug, () => gl.compileShader(vertexShader));
  if (gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS) === false) {
    console.log(gl.getShaderInfoLog(vertexShader));
    throw new Error('Failed to compile vertex shader.');
  }
  return vertexShader;
}

export function createFragmentShader(
    gl: WebGLRenderingContext, debug: boolean,
    fragmentShaderSource: string): WebGLShader {
  const fragmentShader: WebGLShader = throwIfNull<WebGLShader>(
      gl, debug, () => gl.createShader(gl.FRAGMENT_SHADER),
      'Unable to create fragment WebGLShader.');
  callAndCheck(
      gl, debug, () => gl.shaderSource(fragmentShader, fragmentShaderSource));
  callAndCheck(gl, debug, () => gl.compileShader(fragmentShader));
  if (gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS) === false) {
    logShaderSourceAndInfoLog(
        fragmentShaderSource, gl.getShaderInfoLog(fragmentShader));
    throw new Error('Failed to compile fragment shader.');
  }
  return fragmentShader;
}

const lineNumberRegex = /ERROR: [0-9]+:([0-9]+):/g;
function logShaderSourceAndInfoLog(
    shaderSource: string, shaderInfoLog: string) {
  const lineNumberRegexResult = lineNumberRegex.exec(shaderInfoLog);
  if (lineNumberRegexResult == null) {
    console.log(`Couldn't parse line number in error: ${shaderInfoLog}`);
    console.log(shaderSource);
    return;
  }

  const lineNumber = +lineNumberRegexResult[1];

  const shaderLines = shaderSource.split('\n');
  const pad = shaderLines.length.toString().length + 2;
  const linesWithLineNumbers = shaderLines.map(
      (line, lineNumber) =>
          util.rightPad((lineNumber + 1).toString(), pad) + line);
  let maxLineLength = 0;
  for (let i = 0; i < linesWithLineNumbers.length; i++) {
    maxLineLength = Math.max(linesWithLineNumbers[i].length, maxLineLength);
  }

  const beforeErrorLines = linesWithLineNumbers.slice(0, lineNumber - 1);
  const errorLine = linesWithLineNumbers.slice(lineNumber - 1, lineNumber);
  const afterErrorLines = linesWithLineNumbers.slice(lineNumber);

  console.log(beforeErrorLines.join('\n'));
  console.log(shaderInfoLog.split('\n')[0]);
  console.log(
      `%c ${util.rightPad(errorLine[0], maxLineLength)}`,
      'border:1px solid red; background-color:#e3d2d2; color:#a61717');
  console.log(afterErrorLines.join('\n'));
}

export function createProgram(
    gl: WebGLRenderingContext, debug: boolean): WebGLProgram {
  return throwIfNull<WebGLProgram>(
      gl, debug, () => gl.createProgram(), 'Unable to create WebGLProgram.');
}

export function linkProgram(
    gl: WebGLRenderingContext, debug: boolean, program: WebGLProgram) {
  callAndCheck(gl, debug, () => gl.linkProgram(program));
  if (gl.getProgramParameter(program, gl.LINK_STATUS) === false) {
    console.log(gl.getProgramInfoLog(program));
    throw new Error('Failed to link vertex and fragment shaders.');
  }
}

export function validateProgram(
    gl: WebGLRenderingContext, debug: boolean, program: WebGLProgram) {
  callAndCheck(gl, debug, () => gl.validateProgram(program));
  if (gl.getProgramParameter(program, gl.VALIDATE_STATUS) === false) {
    console.log(gl.getProgramInfoLog(program));
    throw new Error('Shader program validation failed.');
  }
}

export function createStaticVertexBuffer(
    gl: WebGLRenderingContext, debug: boolean,
    data: Float32Array): WebGLBuffer {
  const buffer: WebGLBuffer = throwIfNull<WebGLBuffer>(
      gl, debug, () => gl.createBuffer(), 'Unable to create WebGLBuffer');
  callAndCheck(gl, debug, () => gl.bindBuffer(gl.ARRAY_BUFFER, buffer));
  callAndCheck(
      gl, debug, () => gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW));
  return buffer;
}

export function createStaticIndexBuffer(
    gl: WebGLRenderingContext, debug: boolean, data: Uint16Array): WebGLBuffer {
  const buffer: WebGLBuffer = throwIfNull<WebGLBuffer>(
      gl, debug, () => gl.createBuffer(), 'Unable to create WebGLBuffer');
  callAndCheck(gl, debug, () => gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffer));
  callAndCheck(
      gl, debug,
      () => gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data, gl.STATIC_DRAW));
  return buffer;
}

export function getNumChannels(): number {
  if (env().getNumber('WEBGL_VERSION') === 2) {
    return 1;
  }
  return 4;
}

export function createTexture(
    gl: WebGLRenderingContext, debug: boolean): WebGLTexture {
  return throwIfNull<WebGLTexture>(
      gl, debug, () => gl.createTexture(), 'Unable to create WebGLTexture.');
}

export function validateTextureSize(width: number, height: number) {
  const maxTextureSize = env().getNumber('WEBGL_MAX_TEXTURE_SIZE');
  if ((width <= 0) || (height <= 0)) {
    const requested = `[${width}x${height}]`;
    throw new Error('Requested texture size ' + requested + ' is invalid.');
  }
  if ((width > maxTextureSize) || (height > maxTextureSize)) {
    const requested = `[${width}x${height}]`;
    const max = `[${maxTextureSize}x${maxTextureSize}]`;
    throw new Error(
        'Requested texture size ' + requested +
        ' greater than WebGL maximum on this browser / GPU ' + max + '.');
  }
}

export function createFramebuffer(
    gl: WebGLRenderingContext, debug: boolean): WebGLFramebuffer {
  return throwIfNull<WebGLFramebuffer>(
      gl, debug, () => gl.createFramebuffer(),
      'Unable to create WebGLFramebuffer.');
}

export function bindVertexBufferToProgramAttribute(
    gl: WebGLRenderingContext, debug: boolean, program: WebGLProgram,
    attribute: string, buffer: WebGLBuffer, arrayEntriesPerItem: number,
    itemStrideInBytes: number, itemOffsetInBytes: number): boolean {
  const loc = gl.getAttribLocation(program, attribute);
  if (loc === -1) {
    // The GPU compiler decided to strip out this attribute because it's unused,
    // thus no need to bind.
    return false;
  }
  callAndCheck(gl, debug, () => gl.bindBuffer(gl.ARRAY_BUFFER, buffer));
  callAndCheck(
      gl, debug,
      () => gl.vertexAttribPointer(
          loc, arrayEntriesPerItem, gl.FLOAT, false, itemStrideInBytes,
          itemOffsetInBytes));
  callAndCheck(gl, debug, () => gl.enableVertexAttribArray(loc));
  return true;
}

export function bindTextureUnit(
    gl: WebGLRenderingContext, debug: boolean, texture: WebGLTexture,
    textureUnit: number) {
  validateTextureUnit(gl, textureUnit);
  callAndCheck(gl, debug, () => gl.activeTexture(gl.TEXTURE0 + textureUnit));
  callAndCheck(gl, debug, () => gl.bindTexture(gl.TEXTURE_2D, texture));
}

export function unbindTextureUnit(
    gl: WebGLRenderingContext, debug: boolean, textureUnit: number) {
  validateTextureUnit(gl, textureUnit);
  callAndCheck(gl, debug, () => gl.activeTexture(gl.TEXTURE0 + textureUnit));
  callAndCheck(gl, debug, () => gl.bindTexture(gl.TEXTURE_2D, null));
}

export function getProgramUniformLocationOrThrow(
    gl: WebGLRenderingContext, debug: boolean, program: WebGLProgram,
    uniformName: string): WebGLUniformLocation {
  return throwIfNull<WebGLUniformLocation>(
      gl, debug, () => gl.getUniformLocation(program, uniformName),
      'uniform "' + uniformName + '" not present in program.');
}

export function getProgramUniformLocation(
    gl: WebGLRenderingContext, program: WebGLProgram,
    uniformName: string): WebGLUniformLocation {
  return gl.getUniformLocation(program, uniformName);
}

export function bindTextureToProgramUniformSampler(
    gl: WebGLRenderingContext, debug: boolean, program: WebGLProgram,
    texture: WebGLTexture, uniformSamplerLocation: WebGLUniformLocation,
    textureUnit: number) {
  callAndCheck(
      gl, debug, () => bindTextureUnit(gl, debug, texture, textureUnit));
  callAndCheck(
      gl, debug, () => gl.uniform1i(uniformSamplerLocation, textureUnit));
}

export function bindCanvasToFramebuffer(
    gl: WebGLRenderingContext, debug: boolean) {
  callAndCheck(gl, debug, () => gl.bindFramebuffer(gl.FRAMEBUFFER, null));
  callAndCheck(
      gl, debug, () => gl.viewport(0, 0, gl.canvas.width, gl.canvas.height));
  callAndCheck(
      gl, debug, () => gl.scissor(0, 0, gl.canvas.width, gl.canvas.height));
}

export function bindColorTextureToFramebuffer(
    gl: WebGLRenderingContext, debug: boolean, texture: WebGLTexture,
    framebuffer: WebGLFramebuffer) {
  callAndCheck(
      gl, debug, () => gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer));
  callAndCheck(
      gl, debug,
      () => gl.framebufferTexture2D(
          gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0));
}

export function unbindColorTextureFromFramebuffer(
    gl: WebGLRenderingContext, debug: boolean, framebuffer: WebGLFramebuffer) {
  callAndCheck(
      gl, debug, () => gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer));
  callAndCheck(
      gl, debug,
      () => gl.framebufferTexture2D(
          gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0));
}

export function validateFramebuffer(gl: WebGLRenderingContext) {
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    throw new Error(
        'Error binding framebuffer: ' + getFramebufferErrorMessage(gl, status));
  }
}

export function getFramebufferErrorMessage(
    gl: WebGLRenderingContext, status: number): string {
  switch (status) {
    case gl.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
      return 'FRAMEBUFFER_INCOMPLETE_ATTACHMENT';
    case gl.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
      return 'FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT';
    case gl.FRAMEBUFFER_INCOMPLETE_DIMENSIONS:
      return 'FRAMEBUFFER_INCOMPLETE_DIMENSIONS';
    case gl.FRAMEBUFFER_UNSUPPORTED:
      return 'FRAMEBUFFER_UNSUPPORTED';
    default:
      return `unknown error ${status}`;
  }
}

function throwIfNull<T>(
    gl: WebGLRenderingContext, debug: boolean, returnTOrNull: () => T | null,
    failureMessage: string): T {
  const tOrNull: T|null = callAndCheck(gl, debug, () => returnTOrNull());
  if (tOrNull == null) {
    throw new Error(failureMessage);
  }
  return tOrNull;
}

function validateTextureUnit(gl: WebGLRenderingContext, textureUnit: number) {
  const maxTextureUnit = gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS - 1;
  const glTextureUnit = textureUnit + gl.TEXTURE0;
  if (glTextureUnit < gl.TEXTURE0 || glTextureUnit > maxTextureUnit) {
    const textureUnitRange = `[gl.TEXTURE0, gl.TEXTURE${maxTextureUnit}]`;
    throw new Error(`textureUnit must be in ${textureUnitRange}.`);
  }
}

export function getBatchDim(shape: number[], dimsToSkip = 2): number {
  return util.sizeFromShape(shape.slice(0, shape.length - dimsToSkip));
}

export function getRowsCols(shape: number[]): [number, number] {
  if (shape.length === 0) {
    throw Error('Cannot get rows and columns of an empty shape array.');
  }

  return [
    shape.length > 1 ? shape[shape.length - 2] : 1, shape[shape.length - 1]
  ];
}

export function getShapeAs3D(shape: number[]): [number, number, number] {
  let shapeAs3D: [number, number, number] = [1, 1, 1];
  const isScalar = shape.length === 0 || (shape.length === 1 && shape[0] === 1);
  if (!isScalar) {
    shapeAs3D =
        [getBatchDim(shape), ...getRowsCols(shape)] as [number, number, number];
  }
  return shapeAs3D;
}

export function getTextureShapeFromLogicalShape(
    logShape: number[], isPacked = false): [number, number] {
  let maxTexSize = env().getNumber('WEBGL_MAX_TEXTURE_SIZE');
  if (isPacked) {
    maxTexSize = maxTexSize * 2;

    // This logic ensures we accurately count the number of packed texels needed
    // to accommodate the tensor. We can only pack values in the same texel if
    // they are from adjacent pairs of rows/cols within the same batch. So if a
    // tensor has 3 rows, we pretend it has 4 rows in order to account for the
    // fact that the texels containing the third row are half empty.
    logShape = logShape.map(
        (d, i) => i >= logShape.length - 2 ?
            util.nearestLargerEven(logShape[i]) :
            logShape[i]);

    // Packed texture height is at least 2 (the channel height of a single
    // texel).
    if (logShape.length === 1) {
      logShape = [2, logShape[0]];
    }
  }

  // If logical shape is 2, we don't squeeze, since we want to match physical.
  if (logShape.length !== 2) {
    const squeezeResult = util.squeezeShape(logShape);
    logShape = squeezeResult.newShape;
  }

  let size = util.sizeFromShape(logShape);
  if (logShape.length <= 1 && size <= maxTexSize) {
    return [1, size];
  } else if (
      logShape.length === 2 && logShape[0] <= maxTexSize &&
      logShape[1] <= maxTexSize) {
    return logShape as [number, number];
  } else if (
      logShape.length === 3 && logShape[0] * logShape[1] <= maxTexSize &&
      logShape[2] <= maxTexSize) {
    return [logShape[0] * logShape[1], logShape[2]];
  } else if (
      logShape.length === 3 && logShape[0] <= maxTexSize &&
      logShape[1] * logShape[2] <= maxTexSize) {
    return [logShape[0], logShape[1] * logShape[2]];
  } else if (
      logShape.length === 4 &&
      logShape[0] * logShape[1] * logShape[2] <= maxTexSize &&
      logShape[3] <= maxTexSize) {
    return [logShape[0] * logShape[1] * logShape[2], logShape[3]];
  } else if (
      logShape.length === 4 && logShape[0] <= maxTexSize &&
      logShape[1] * logShape[2] * logShape[3] <= maxTexSize) {
    return [logShape[0], logShape[1] * logShape[2] * logShape[3]];
  } else {
    if (isPacked) {
      // For packed textures size equals the number of channels required to
      // accommodate the texture data. However in order to squarify such that
      // inner dimensions stay even, we rewrite size to equal the number of
      // texels. Then in the return statement we rehydrate the squarified
      // dimensions to channel units.

      const batchDim = getBatchDim(logShape);
      let rows = 2, cols = 2;
      if (logShape.length) {
        [rows, cols] = getRowsCols(logShape);
      }
      size = batchDim * (rows / 2) * (cols / 2);
      return util.sizeToSquarishShape(size).map(d => d * 2) as [number, number];
    }
    return util.sizeToSquarishShape(size);
  }
}

function isEven(n: number): boolean {
  return n % 2 === 0;
}

/**
 * This determines whether reshaping a packed texture requires rearranging
 * the data within the texture, assuming 2x2 packing.
 */
export function isReshapeFree(shape1: number[], shape2: number[]): boolean {
  shape1 = shape1.slice(-2);
  shape2 = shape2.slice(-2);

  if (util.arraysEqual(shape1, shape2)) {
    return true;
  }

  if (!shape1.length || !shape2.length) {  // One of the shapes is a scalar.
    return true;
  }

  if (shape1[0] === 0 || shape1[1] === 0 || shape2[0] === 0 ||
      shape2[1] === 0) {
    return true;
  }

  if (shape1.length !== shape2.length) {  // One of the shapes is a vector.
    const shape1Cols = shape1.slice(-1)[0];
    const shape2Cols = shape2.slice(-1)[0];
    if (shape1Cols === shape2Cols) {
      return true;
    }

    if (isEven(shape1Cols) && isEven(shape2Cols) &&
        (shape1[0] === 1 || shape2[0] === 1)) {
      return true;
    }
  }
  return shape1[1] === shape2[1] && isEven(shape1[0]) && isEven(shape2[0]);
}

// We cache webgl params because the environment gets reset between
// unit tests and we don't want to constantly query the WebGLContext for
// MAX_TEXTURE_SIZE.
let MAX_TEXTURE_SIZE: number;
let MAX_TEXTURES_IN_SHADER: number;

export function getWebGLMaxTextureSize(webGLVersion: number): number {
  if (MAX_TEXTURE_SIZE == null) {
    const gl = getWebGLContext(webGLVersion);
    MAX_TEXTURE_SIZE = gl.getParameter(gl.MAX_TEXTURE_SIZE);
  }
  return MAX_TEXTURE_SIZE;
}

export function resetMaxTextureSize() {
  MAX_TEXTURE_SIZE = null;
}
export function resetMaxTexturesInShader() {
  MAX_TEXTURES_IN_SHADER = null;
}

export function getMaxTexturesInShader(webGLVersion: number): number {
  if (MAX_TEXTURES_IN_SHADER == null) {
    const gl = getWebGLContext(webGLVersion);
    MAX_TEXTURES_IN_SHADER = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS);
  }
  // We cap at 16 to avoid spurious runtime "memory exhausted" error.
  return Math.min(16, MAX_TEXTURES_IN_SHADER);
}

export function getWebGLDisjointQueryTimerVersion(webGLVersion: number):
    number {
  if (webGLVersion === 0) {
    return 0;
  }

  let queryTimerVersion: number;
  const gl = getWebGLContext(webGLVersion);

  if (hasExtension(gl, 'EXT_disjoint_timer_query_webgl2') &&
      webGLVersion === 2) {
    queryTimerVersion = 2;
  } else if (hasExtension(gl, 'EXT_disjoint_timer_query')) {
    queryTimerVersion = 1;
  } else {
    queryTimerVersion = 0;
  }
  return queryTimerVersion;
}

export function hasExtension(gl: WebGLRenderingContext, extensionName: string) {
  const ext = gl.getExtension(extensionName);
  return ext != null;
}

export function isWebGLVersionEnabled(webGLVersion: 1|2) {
  try {
    const gl = getWebGLContext(webGLVersion);
    if (gl != null) {
      return true;
    }
  } catch (e) {
    return false;
  }
  return false;
}

export function isCapableOfRenderingToFloatTexture(webGLVersion: number):
    boolean {
  if (webGLVersion === 0) {
    return false;
  }

  const gl = getWebGLContext(webGLVersion);

  if (webGLVersion === 1) {
    if (!hasExtension(gl, 'OES_texture_float')) {
      return false;
    }
  } else {
    if (!hasExtension(gl, 'EXT_color_buffer_float')) {
      return false;
    }
  }

  const isFrameBufferComplete = createFloatTextureAndBindToFramebuffer(gl);
  return isFrameBufferComplete;
}

/**
 * Check if we can download values from a float/half-float texture.
 *
 * Note that for performance reasons we use binding a texture to a framebuffer
 * as a proxy for ability to download float values later using readPixels. The
 * texture params of this texture will not match those in readPixels exactly
 * but if we are unable to bind some kind of float texture to the frameBuffer
 * then we definitely will not be able to read float values from it.
 */
export function isDownloadFloatTextureEnabled(webGLVersion: number): boolean {
  if (webGLVersion === 0) {
    return false;
  }

  const gl = getWebGLContext(webGLVersion);

  if (webGLVersion === 1) {
    if (!hasExtension(gl, 'OES_texture_float')) {
      return false;
    }
    if (!hasExtension(gl, 'WEBGL_color_buffer_float')) {
      return false;
    }
  } else {
    if (hasExtension(gl, 'EXT_color_buffer_float')) {
      return createFloatTextureAndBindToFramebuffer(gl);
    }

    const COLOR_BUFFER_HALF_FLOAT = 'EXT_color_buffer_half_float';
    if (hasExtension(gl, COLOR_BUFFER_HALF_FLOAT)) {
      const textureHalfFloatExtension =
          gl.getExtension(COLOR_BUFFER_HALF_FLOAT);
      return createHalfFloatTextureAndBindToFramebuffer(
          gl, textureHalfFloatExtension);
    }

    return false;
  }

  const isFrameBufferComplete = createFloatTextureAndBindToFramebuffer(gl);
  return isFrameBufferComplete;
}

function createFloatTextureAndBindToFramebuffer(gl: WebGLRenderingContext):
    boolean {
  const texConfig = getTextureConfig(gl);

  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);

  const width = 1;
  const height = 1;
  gl.texImage2D(
      gl.TEXTURE_2D, 0, texConfig.internalFormatFloat, width, height, 0,
      texConfig.textureFormatFloat, texConfig.textureTypeFloat, null);

  const frameBuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
  gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

  const isFrameBufferComplete =
      gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;

  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.deleteTexture(texture);
  gl.deleteFramebuffer(frameBuffer);

  return isFrameBufferComplete;
}

function createHalfFloatTextureAndBindToFramebuffer(
    // tslint:disable-next-line:no-any
    gl: WebGLRenderingContext, textureHalfFloatExtension: any): boolean {
  const texConfig = getTextureConfig(gl, textureHalfFloatExtension);
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);

  const width = 1;
  const height = 1;
  gl.texImage2D(
      gl.TEXTURE_2D, 0, texConfig.internalFormatHalfFloat, width, height, 0,
      texConfig.textureFormatFloat, texConfig.textureTypeHalfFloat, null);

  const frameBuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
  gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

  const isFrameBufferComplete =
      gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;

  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.deleteTexture(texture);
  gl.deleteFramebuffer(frameBuffer);

  return isFrameBufferComplete;
}

export function isWebGLFenceEnabled(webGLVersion: number) {
  if (webGLVersion !== 2) {
    return false;
  }
  const gl = getWebGLContext(webGLVersion);

  // tslint:disable-next-line:no-any
  const isEnabled = (gl as any).fenceSync != null;
  return isEnabled;
}
