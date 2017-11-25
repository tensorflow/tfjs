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

let MAX_TEXTURE_SIZE: number = null;

import * as util from '../../../util';
import {ENV} from '../../../environment';

export interface WebGLContextAttributes {
  alpha?: boolean;
  antialias?: boolean;
  premultipliedAlpha?: boolean;
  preserveDrawingBuffer?: boolean;
  depth?: boolean;
  stencil?: boolean;
  failIfMajorPerformanceCaveat?: boolean;
}

export interface WebGLLoseContextExtension { loseContext(): void; }

export function createWebGLRenderingContext(attributes: WebGLContextAttributes):
    WebGLRenderingContext {
  const canvas = document.createElement('canvas');
  canvas.width = 1;
  canvas.height = 1;
  return createWebGLRenderingContextFromCanvas(canvas, attributes);
}

export function createWebGLRenderingContextFromCanvas(
    canvas: HTMLCanvasElement,
    attributes: WebGLContextAttributes): WebGLRenderingContext {
  let gl: WebGLRenderingContext;

  const webglVersion = ENV.get('WEBGL_VERSION');
  if (webglVersion === 2) {
    gl = canvas.getContext('webgl2', attributes) as WebGLRenderingContext;
  } else if (webglVersion === 1) {
    gl = (canvas.getContext('webgl', attributes) ||
          canvas.getContext('experimental-webgl', attributes)) as
        WebGLRenderingContext;
  }

  if (webglVersion === 0 || gl == null) {
    throw new Error('This browser does not support WebGL.');
  }
  return gl;
}

export function callAndCheck<T>(gl: WebGLRenderingContext, func: () => T): T {
  const returnValue = func();
  checkWebGLError(gl);
  return returnValue;
}

let webGLDebugErrorCheckingEnabled = false;

export function enableDebugWebGLErrorChecking(enabled: boolean) {
  webGLDebugErrorCheckingEnabled = enabled;
}

export function checkWebGLError(gl: WebGLRenderingContext) {
  if (webGLDebugErrorCheckingEnabled) {
    const error = gl.getError();
    if (error !== gl.NO_ERROR) {
      throw new Error('WebGL Error: ' + getWebGLErrorMessage(gl, error));
    }
  }
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
    gl: WebGLRenderingContext, extensionName: string): {} {
  return throwIfNull<{}>(
      gl, () => gl.getExtension(extensionName),
      'Extension "' + extensionName + '" not supported on this browser.');
}

export function createVertexShader(
    gl: WebGLRenderingContext, vertexShaderSource: string): WebGLShader {
  const vertexShader: WebGLShader = throwIfNull<WebGLShader>(
      gl, () => gl.createShader(gl.VERTEX_SHADER),
      'Unable to create vertex WebGLShader.');
  callAndCheck(gl, () => gl.shaderSource(vertexShader, vertexShaderSource));
  callAndCheck(gl, () => gl.compileShader(vertexShader));
  if (gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS) === false) {
    console.log(gl.getShaderInfoLog(vertexShader));
    throw new Error('Failed to compile vertex shader.');
  }
  return vertexShader;
}

export function createFragmentShader(
    gl: WebGLRenderingContext, fragmentShaderSource: string): WebGLShader {
  const fragmentShader: WebGLShader = throwIfNull<WebGLShader>(
      gl, () => gl.createShader(gl.FRAGMENT_SHADER),
      'Unable to create fragment WebGLShader.');
  callAndCheck(gl, () => gl.shaderSource(fragmentShader, fragmentShaderSource));
  callAndCheck(gl, () => gl.compileShader(fragmentShader));
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

export function createProgram(gl: WebGLRenderingContext): WebGLProgram {
  return throwIfNull<WebGLProgram>(
      gl, () => gl.createProgram(), 'Unable to create WebGLProgram.');
}

export function linkProgram(gl: WebGLRenderingContext, program: WebGLProgram) {
  callAndCheck(gl, () => gl.linkProgram(program));
  if (gl.getProgramParameter(program, gl.LINK_STATUS) === false) {
    console.log(gl.getProgramInfoLog(program));
    throw new Error('Failed to link vertex and fragment shaders.');
  }
}

export function validateProgram(
    gl: WebGLRenderingContext, program: WebGLProgram) {
  callAndCheck(gl, () => gl.validateProgram(program));
  if (gl.getProgramParameter(program, gl.VALIDATE_STATUS) === false) {
    console.log(gl.getProgramInfoLog(program));
    throw new Error('Shader program validation failed.');
  }
}

export function createStaticVertexBuffer(
    gl: WebGLRenderingContext, data: Float32Array): WebGLBuffer {
  const buffer: WebGLBuffer = throwIfNull<WebGLBuffer>(
      gl, () => gl.createBuffer(), 'Unable to create WebGLBuffer');
  callAndCheck(gl, () => gl.bindBuffer(gl.ARRAY_BUFFER, buffer));
  callAndCheck(gl, () => gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW));
  return buffer;
}

export function createStaticIndexBuffer(
    gl: WebGLRenderingContext, data: Uint16Array): WebGLBuffer {
  const buffer: WebGLBuffer = throwIfNull<WebGLBuffer>(
      gl, () => gl.createBuffer(), 'Unable to create WebGLBuffer');
  callAndCheck(gl, () => gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffer));
  callAndCheck(
      gl, () => gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data, gl.STATIC_DRAW));
  return buffer;
}

export function queryMaxTextureSize(gl: WebGLRenderingContext): number {
  if (MAX_TEXTURE_SIZE != null) {
    return MAX_TEXTURE_SIZE;
  }
  MAX_TEXTURE_SIZE =
      callAndCheck(gl, () => gl.getParameter(gl.MAX_TEXTURE_SIZE));
  return MAX_TEXTURE_SIZE;
}

export function getChannelsPerTexture(): number {
  if (!ENV.get('WEBGL_FLOAT_TEXTURE_ENABLED')) {
    return 4;
  }

  if (ENV.get('WEBGL_VERSION') === 2) {
    return 1;
  }
  return 4;
}

export function createTexture(gl: WebGLRenderingContext): WebGLTexture {
  return throwIfNull<WebGLTexture>(
      gl, () => gl.createTexture(), 'Unable to create WebGLTexture.');
}

export function validateTextureSize(
    gl: WebGLRenderingContext, width: number, height: number) {
  const maxTextureSize: number = queryMaxTextureSize(gl);
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

export function createFramebuffer(gl: WebGLRenderingContext): WebGLFramebuffer {
  return throwIfNull<WebGLFramebuffer>(
      gl, () => gl.createFramebuffer(), 'Unable to create WebGLFramebuffer.');
}

export function bindVertexBufferToProgramAttribute(
    gl: WebGLRenderingContext, program: WebGLProgram, attribute: string,
    buffer: WebGLBuffer, arrayEntriesPerItem: number, itemStrideInBytes: number,
    itemOffsetInBytes: number, attribLocations?: {[name: string]: number}) {
  let loc = -1;
  if ((attribLocations != null) && (attribute in attribLocations)) {
    loc = attribLocations[attribute];
  } else {
    loc = gl.getAttribLocation(program, attribute);
  }
  if (loc === -1) {
    // The GPU compiler decided to strip out this attribute because it's unused,
    // thus no need to bind.
    return;
  }
  callAndCheck(gl, () => gl.bindBuffer(gl.ARRAY_BUFFER, buffer));
  callAndCheck(
      gl,
      () => gl.vertexAttribPointer(
          loc, arrayEntriesPerItem, gl.FLOAT, false, itemStrideInBytes,
          itemOffsetInBytes));
  callAndCheck(gl, () => gl.enableVertexAttribArray(loc));
}

export function bindTextureUnit(
    gl: WebGLRenderingContext, texture: WebGLTexture, textureUnit: number) {
  validateTextureUnit(gl, textureUnit);
  callAndCheck(gl, () => gl.activeTexture(gl.TEXTURE0 + textureUnit));
  callAndCheck(gl, () => gl.bindTexture(gl.TEXTURE_2D, texture));
}

export function unbindTextureUnit(
    gl: WebGLRenderingContext, textureUnit: number) {
  validateTextureUnit(gl, textureUnit);
  callAndCheck(gl, () => gl.activeTexture(gl.TEXTURE0 + textureUnit));
  callAndCheck(gl, () => gl.bindTexture(gl.TEXTURE_2D, null));
}

export function getProgramUniformLocationOrThrow(
    gl: WebGLRenderingContext, program: WebGLProgram,
    uniformName: string): WebGLUniformLocation {
  return throwIfNull<WebGLUniformLocation>(
      gl, () => gl.getUniformLocation(program, uniformName),
      'uniform "' + uniformName + '" not present in program.');
}

export function bindTextureToProgramUniformSampler(
    gl: WebGLRenderingContext, program: WebGLProgram, texture: WebGLTexture,
    uniformSamplerLocation: WebGLUniformLocation, textureUnit: number) {
  callAndCheck(gl, () => bindTextureUnit(gl, texture, textureUnit));
  callAndCheck(gl, () => gl.uniform1i(uniformSamplerLocation, textureUnit));
}

export function bindCanvasToFramebuffer(gl: WebGLRenderingContext) {
  callAndCheck(gl, () => gl.bindFramebuffer(gl.FRAMEBUFFER, null));
  callAndCheck(gl, () => gl.viewport(0, 0, gl.canvas.width, gl.canvas.height));
  callAndCheck(gl, () => gl.scissor(0, 0, gl.canvas.width, gl.canvas.height));
}

export function bindColorTextureToFramebuffer(
    gl: WebGLRenderingContext, texture: WebGLTexture,
    framebuffer: WebGLFramebuffer) {
  callAndCheck(gl, () => gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer));
  callAndCheck(
      gl,
      () => gl.framebufferTexture2D(
          gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0));
}

export function unbindColorTextureFromFramebuffer(
    gl: WebGLRenderingContext, framebuffer: WebGLFramebuffer) {
  callAndCheck(gl, () => gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer));
  callAndCheck(
      gl,
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
    gl: WebGLRenderingContext, returnTOrNull: () => T | null,
    failureMessage: string): T {
  const tOrNull: T|null = callAndCheck(gl, () => returnTOrNull());
  if (tOrNull == null) {
    throw new Error(failureMessage);
  }
  return tOrNull as T;
}

function validateTextureUnit(gl: WebGLRenderingContext, textureUnit: number) {
  const maxTextureUnit = gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS - 1;
  const glTextureUnit = textureUnit + gl.TEXTURE0;
  if (glTextureUnit < gl.TEXTURE0 || glTextureUnit > maxTextureUnit) {
    const textureUnitRange = `[gl.TEXTURE0, gl.TEXTURE${maxTextureUnit}]`;
    throw new Error(`textureUnit must be in ${textureUnitRange}.`);
  }
}

export function getTextureShapeFromLogicalShape(
    gl: WebGLRenderingContext, logShape: number[]): [number, number] {
  // If logical shape is 2, we don't squeeze, since we want to match physical.
  if (logShape.length !== 2) {
    const squeezeResult = util.squeezeShape(logShape);
    logShape = squeezeResult.newShape;
  }

  const maxTexSize = queryMaxTextureSize(gl);
  const size = util.sizeFromShape(logShape);
  if (logShape.length <= 1 && size <= maxTexSize) {
    return [size, 1];
  } else if (
      logShape.length === 2 && logShape[0] <= maxTexSize &&
      logShape[1] <= maxTexSize) {
    return logShape as [number, number];
  } else if (
      logShape.length === 3 && logShape[0] <= maxTexSize &&
      logShape[1] * logShape[2] <= maxTexSize) {
    return [logShape[0], logShape[1] * logShape[2]];
  } else if (
      logShape.length === 4 && logShape[0] <= maxTexSize &&
      logShape[1] * logShape[2] * logShape[3] <= maxTexSize) {
    return [logShape[0], logShape[1] * logShape[2] * logShape[3]];
  } else {
    return util.sizeToSquarishShape(size);
  }
}
