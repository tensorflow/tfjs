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

import {ENV} from '../../environment';
import * as util from '../../util';
import {getGlslDifferences} from './glsl_version';
import * as tex_util from './tex_util';
import * as webgl_util from './webgl_util';

export interface TextureConfig {
  internalFormatFloat: number;
  textureFormatFloat: number;
  internalFormatPackedHalfFloat: number;
  internalFormatHalfFloat: number;
  internalFormatPackedFloat: number;

  // The format to use during a gl.readPixels call.
  downloadTextureFormat: number;
  // How many channels need to be unpacked after a gl.readPixels call.
  downloadUnpackNumChannels: number;

  defaultNumChannels: number;
  textureTypeHalfFloat: number;
}

export function createVertexShader(
    gl: WebGLRenderingContext, debug: boolean): WebGLShader {
  const glsl = getGlslDifferences();
  const vertexShaderSource = `${glsl.version}
    precision highp float;
    ${glsl.attribute} vec3 clipSpacePos;
    ${glsl.attribute} vec2 uv;
    ${glsl.varyingVs} vec2 resultUV;

    void main() {
      gl_Position = vec4(clipSpacePos, 1);
      resultUV = uv;
    }`;
  return webgl_util.createVertexShader(gl, debug, vertexShaderSource);
}

export function createVertexBuffer(
    gl: WebGLRenderingContext, debug: boolean): WebGLBuffer {
  // [x y z u v] * [upper-left, lower-left, upper-right, lower-right]
  const vertexArray = new Float32Array(
      [-1, 1, 0, 0, 1, -1, -1, 0, 0, 0, 1, 1, 0, 1, 1, 1, -1, 0, 1, 0]);
  return webgl_util.createStaticVertexBuffer(gl, debug, vertexArray);
}

export function createIndexBuffer(
    gl: WebGLRenderingContext, debug: boolean): WebGLBuffer {
  // OpenGL (and WebGL) have "CCW == front" winding
  const triangleVertexIndices = new Uint16Array([0, 1, 2, 2, 1, 3]);
  return webgl_util.createStaticIndexBuffer(gl, debug, triangleVertexIndices);
}

export function getTextureConfig(
    // tslint:disable-next-line:no-any
    gl: WebGLRenderingContext, textureHalfFloatExtension?: any): TextureConfig {
  // tslint:disable-next-line:no-any
  const glany = gl as any;

  let internalFormatFloat: number;
  let internalFormatHalfFloat: number;
  let internalFormatPackedHalfFloat: number;
  let internalFormatPackedFloat: number;
  let textureFormatFloat: number;

  let downloadTextureFormat: number;
  let downloadUnpackNumChannels: number;

  let defaultNumChannels: number;
  let textureTypeHalfFloat: number;

  if (ENV.get('WEBGL_VERSION') === 2) {
    internalFormatFloat = glany.R32F;
    internalFormatHalfFloat = glany.R16F;
    internalFormatPackedHalfFloat = glany.RGBA16F;
    internalFormatPackedFloat = glany.RGBA32F;
    textureFormatFloat = glany.RED;
    downloadUnpackNumChannels = 4;
    defaultNumChannels = 1;
    textureTypeHalfFloat = glany.HALF_FLOAT;
  } else {
    internalFormatFloat = gl.RGBA;
    internalFormatHalfFloat = gl.RGBA;
    internalFormatPackedHalfFloat = gl.RGBA;
    internalFormatPackedFloat = glany.RGBA;
    textureFormatFloat = gl.RGBA;
    downloadUnpackNumChannels = 4;
    defaultNumChannels = 4;
    textureTypeHalfFloat = textureHalfFloatExtension != null ?
        textureHalfFloatExtension.HALF_FLOAT_OES :
        null;
  }
  downloadTextureFormat = gl.RGBA;

  return {
    internalFormatFloat,
    internalFormatHalfFloat,
    internalFormatPackedHalfFloat,
    internalFormatPackedFloat,
    textureFormatFloat,
    downloadTextureFormat,
    downloadUnpackNumChannels,
    defaultNumChannels,
    textureTypeHalfFloat
  };
}

function createAndConfigureTexture(
    gl: WebGLRenderingContext, debug: boolean, width: number, height: number,
    internalFormat: number, textureFormat: number,
    textureType: number): WebGLTexture {
  webgl_util.validateTextureSize(width, height);
  const texture = webgl_util.createTexture(gl, debug);

  const tex2d = gl.TEXTURE_2D;
  webgl_util.callAndCheck(gl, debug, () => gl.bindTexture(tex2d, texture));
  webgl_util.callAndCheck(
      gl, debug,
      () => gl.texParameteri(tex2d, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE));
  webgl_util.callAndCheck(
      gl, debug,
      () => gl.texParameteri(tex2d, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE));
  webgl_util.callAndCheck(
      gl, debug,
      () => gl.texParameteri(tex2d, gl.TEXTURE_MIN_FILTER, gl.NEAREST));
  webgl_util.callAndCheck(
      gl, debug,
      () => gl.texParameteri(tex2d, gl.TEXTURE_MAG_FILTER, gl.NEAREST));
  webgl_util.callAndCheck(
      gl, debug,
      () => gl.texImage2D(
          tex2d, 0, internalFormat, width, height, 0, textureFormat,
          textureType, null));
  webgl_util.callAndCheck(gl, debug, () => gl.bindTexture(gl.TEXTURE_2D, null));
  return texture;
}

export function createFloat32MatrixTexture(
    gl: WebGLRenderingContext, debug: boolean, rows: number, columns: number,
    textureConfig: TextureConfig): WebGLTexture {
  const [width, height] =
      tex_util.getUnpackedMatrixTextureShapeWidthHeight(rows, columns);
  return createAndConfigureTexture(
      gl, debug, width, height, textureConfig.internalFormatFloat,
      textureConfig.textureFormatFloat, gl.FLOAT);
}

export function createFloat16MatrixTexture(
    gl: WebGLRenderingContext, debug: boolean, rows: number, columns: number,
    textureConfig: TextureConfig): WebGLTexture {
  const [width, height] =
      tex_util.getUnpackedMatrixTextureShapeWidthHeight(rows, columns);
  return createAndConfigureTexture(
      gl, debug, width, height, textureConfig.internalFormatHalfFloat,
      textureConfig.textureFormatFloat, textureConfig.textureTypeHalfFloat);
}

export function createUnsignedBytesMatrixTexture(
    gl: WebGLRenderingContext, debug: boolean, rows: number, columns: number,
    textureConfig: TextureConfig): WebGLTexture {
  const [width, height] =
      tex_util.getUnpackedMatrixTextureShapeWidthHeight(rows, columns);
  return createAndConfigureTexture(
      gl, debug, width, height, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE);
}

export function createPackedMatrixTexture(
    gl: WebGLRenderingContext, debug: boolean, rows: number, columns: number,
    textureConfig: TextureConfig): WebGLTexture {
  const [width, height] =
      tex_util.getPackedMatrixTextureShapeWidthHeight(rows, columns);
  return createAndConfigureTexture(
      gl, debug, width, height, textureConfig.internalFormatPackedFloat,
      gl.RGBA, gl.FLOAT);
}

export function createFloat16PackedMatrixTexture(
    gl: WebGLRenderingContext, debug: boolean, rows: number, columns: number,
    textureConfig: TextureConfig): WebGLTexture {
  const [width, height] =
      tex_util.getPackedMatrixTextureShapeWidthHeight(rows, columns);
  return createAndConfigureTexture(
      gl, debug, width, height, textureConfig.internalFormatPackedHalfFloat,
      gl.RGBA, textureConfig.textureTypeHalfFloat);
}

export function bindVertexProgramAttributeStreams(
    gl: WebGLRenderingContext, debug: boolean, program: WebGLProgram,
    vertexBuffer: WebGLBuffer): boolean {
  const posOffset = 0;               // x is the first buffer element
  const uvOffset = 3 * 4;            // uv comes after [x y z]
  const stride = (3 * 4) + (2 * 4);  // xyz + uv, each entry is 4-byte float.
  webgl_util.callAndCheck(
      gl, debug, () => gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer));
  const success = webgl_util.bindVertexBufferToProgramAttribute(
      gl, debug, program, 'clipSpacePos', vertexBuffer, 3, stride, posOffset);
  return success &&
      webgl_util.bindVertexBufferToProgramAttribute(
          gl, debug, program, 'uv', vertexBuffer, 2, stride, uvOffset);
}

export function uploadPixelDataToTexture(
    gl: WebGLRenderingContext, debug: boolean, texture: WebGLTexture,
    pixels: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement) {
  webgl_util.callAndCheck(
      gl, debug, () => gl.bindTexture(gl.TEXTURE_2D, texture));
  webgl_util.callAndCheck(
      gl, debug,
      () => gl.texImage2D(
          gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, pixels));
  webgl_util.callAndCheck(gl, debug, () => gl.bindTexture(gl.TEXTURE_2D, null));
}

function uploadDataToTexture(
    gl: WebGLRenderingContext, debug: boolean, texture: WebGLTexture,
    width: number, height: number, data: Float32Array, textureFormat: number) {
  webgl_util.validateTextureSize(width, height);
  webgl_util.callAndCheck(
      gl, debug, () => gl.bindTexture(gl.TEXTURE_2D, texture));
  webgl_util.callAndCheck(
      gl, debug,
      () => gl.texSubImage2D(
          gl.TEXTURE_2D, 0, 0, 0, width, height, textureFormat, gl.FLOAT,
          data));

  webgl_util.callAndCheck(gl, debug, () => gl.bindTexture(gl.TEXTURE_2D, null));
}

export function uploadMatrixToTexture(
    gl: WebGLRenderingContext, debug: boolean, texture: WebGLTexture,
    rows: number, columns: number, matrix: Float32Array, numChannels: number,
    textureConfig: TextureConfig) {
  const [w, h] =
      tex_util.getUnpackedMatrixTextureShapeWidthHeight(rows, columns);

  let unpackedArray: Float32Array;
  const numTexels = rows * columns;
  if (textureConfig.defaultNumChannels === 1 && numTexels === matrix.length) {
    // No need to allocate a temporary array.
    unpackedArray = matrix;
  } else {
    unpackedArray = new Float32Array(numTexels * numChannels);
    tex_util.encodeMatrixToUnpackedArray(matrix, unpackedArray, numChannels);
  }

  uploadDataToTexture(
      gl, debug, texture, w, h, unpackedArray,
      textureConfig.textureFormatFloat);
}

/**
 * This method writes a tensor to a packed texture in a way that respects how we
 * represent data using each texel's r,g,b,a channels. Specifically, we lay
 * out the four channels in two rows each containing two channels, so a single
 * texel can represent up to four values from the tensor. That means a texture
 * that has a channel width of 11 and channel height of 4 will have a texel
 * width of 6 and texel height of 2.
 *
 * rows, columns: Logical number of rows and columns in the tensor to be
 * uploaded.
 *
 * physicalRows, physicalCols: Channel dimensions of the texture that will hold
 * the tensor.
 *
 * width, height (internal parameters): Texel dimensions of the texture.
 */
export function uploadMatrixToPackedTexture(
    gl: WebGLRenderingContext, debug: boolean, texture: WebGLTexture,
    batch: number, rows: number, columns: number, physicalRows: number,
    physicalCols: number, matrix: Float32Array, textureConfig: TextureConfig) {
  const [w, h] = tex_util.getPackedMatrixTextureShapeWidthHeight(
      physicalRows, physicalCols);
  const packedRGBA =
      new Float32Array(tex_util.getPackedRGBAArraySizeFromMatrixShape(
          physicalRows, physicalCols));
  tex_util.encodeMatrixToPackedRGBA(matrix, batch, rows, columns, packedRGBA);
  uploadDataToTexture(gl, debug, texture, w, h, packedRGBA, gl.RGBA);
}

export function maybeCreateBufferFromOutputTexture(
    gl: WebGLRenderingContext, debug: boolean, texture: WebGLTexture,
    rows: number, columns: number, textureConfig: TextureConfig): WebGLBuffer|
    WebGLTexture {
  let bufferOrTexture: WebGLBuffer|WebGLTexture = texture;

  if (ENV.get('WEBGL_VERSION') === 2) {
    const gl2 = gl as WebGL2RenderingContext;

    // Create and bind the buffer.
    const buffer = gl2.createBuffer();
    webgl_util.callAndCheck(
        gl, debug, () => gl.bindBuffer(gl2.PIXEL_PACK_BUFFER, buffer));

    // Initialize the buffer to the size of the texture in bytes.
    const bytesPerFloat = 4;
    const bufferSizeBytes = bytesPerFloat *
        tex_util.getUnpackedArraySizeFromMatrixSize(
            rows * columns, textureConfig.downloadUnpackNumChannels);

    webgl_util.callAndCheck(
        gl, debug,
        () => gl.bufferData(
            gl2.PIXEL_PACK_BUFFER, bufferSizeBytes, gl2.STREAM_READ));

    // Enqueue a command on the GPU command queue to copy of texture into the
    // buffer.
    webgl_util.callAndCheck(
        gl, debug,
        () => gl2.readPixels(0, 0, columns, rows, gl.RGBA, gl.FLOAT, 0));

    webgl_util.callAndCheck(
        gl, debug, () => gl.bindBuffer(gl2.PIXEL_PACK_BUFFER, null));

    bufferOrTexture = buffer;
  }

  return bufferOrTexture;
}

export function downloadFloat32MatrixFromBuffer(
    gl: WebGLRenderingContext, buffer: WebGLBuffer, rows: number,
    columns: number, textureConfig: TextureConfig): Float32Array {
  const gl2 = gl as WebGL2RenderingContext;

  const downloadTarget =
      new Float32Array(tex_util.getUnpackedArraySizeFromMatrixSize(
          rows * columns, textureConfig.downloadUnpackNumChannels));

  gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, buffer);
  gl2.getBufferSubData(gl2.PIXEL_PACK_BUFFER, 0, downloadTarget);
  gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, null);

  const matrix = new Float32Array(rows * columns);
  tex_util.decodeMatrixFromUnpackedArray(
      downloadTarget as Float32Array, matrix,
      textureConfig.downloadUnpackNumChannels);

  return matrix;
}

export function downloadFloat32MatrixFromOutputTexture(
    gl: WebGLRenderingContext, debug: boolean, rows: number, columns: number,
    textureConfig: TextureConfig): Float32Array {
  const [w, h] =
      tex_util.getUnpackedMatrixTextureShapeWidthHeight(rows, columns);

  const downloadTarget =
      new Float32Array(tex_util.getUnpackedArraySizeFromMatrixSize(
          rows * columns, textureConfig.downloadUnpackNumChannels));

  webgl_util.callAndCheck(
      gl, debug,
      () => gl.readPixels(
          0, 0, w, h, textureConfig.downloadTextureFormat, gl.FLOAT,
          downloadTarget));

  const matrix = new Float32Array(rows * columns);
  tex_util.decodeMatrixFromUnpackedArray(
      downloadTarget as Float32Array, matrix,
      textureConfig.downloadUnpackNumChannels);
  return matrix;
}

export function downloadByteEncodedFloatMatrixFromOutputTexture(
    gl: WebGLRenderingContext, debug: boolean, rows: number, columns: number,
    textureConfig: TextureConfig) {
  const [w, h] =
      tex_util.getUnpackedMatrixTextureShapeWidthHeight(rows, columns);

  const numChannels = 4;
  const downloadTarget = new Uint8Array(
      tex_util.getUnpackedArraySizeFromMatrixSize(rows * columns, numChannels));

  webgl_util.callAndCheck(
      gl, debug,
      () => gl.readPixels(
          0, 0, w, h, textureConfig.downloadTextureFormat, gl.UNSIGNED_BYTE,
          downloadTarget));

  // By wrapping the buffer in a Float32Array, we use native browser IEEE 754
  // decoding of the 4 bytes that back each 32 bit float.
  return new Float32Array(downloadTarget.buffer);
}

export function downloadPackedMatrixFromBuffer(
    gl: WebGLRenderingContext, buffer: WebGLBuffer, batch: number, rows: number,
    cols: number, physicalRows: number, physicalCols: number,
    textureConfig: TextureConfig): Float32Array {
  const gl2 = gl as WebGL2RenderingContext;

  const downloadTarget =
      new Float32Array(tex_util.getPackedRGBAArraySizeFromMatrixShape(
          physicalRows, physicalCols));

  gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, buffer);
  gl2.getBufferSubData(gl2.PIXEL_PACK_BUFFER, 0, downloadTarget);
  gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, null);

  const matrix = new Float32Array(util.sizeFromShape([batch, rows, cols]));
  tex_util.decodeMatrixFromPackedRGBA(
      downloadTarget, batch, rows, cols, matrix);
  return matrix;
}

export function downloadMatrixFromPackedOutputTexture(
    gl: WebGLRenderingContext, debug: boolean, batch: number, rows: number,
    cols: number, physicalRows: number, physicalCols: number,
    textureConfig: TextureConfig): Float32Array {
  const [w, h] = tex_util.getPackedMatrixTextureShapeWidthHeight(
      physicalRows, physicalCols);

  const packedRGBA =
      new Float32Array(tex_util.getPackedRGBAArraySizeFromMatrixShape(
          physicalRows, physicalCols));
  webgl_util.callAndCheck(
      gl, debug,
      () => gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, packedRGBA));
  const matrix = new Float32Array(util.sizeFromShape([batch, rows, cols]));
  return tex_util.decodeMatrixFromPackedRGBA(
      packedRGBA, batch, rows, cols, matrix);
}
