/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

import {PixelData, TypedArray} from '@tensorflow/tfjs-core';

import {getGlslDifferences} from './glsl_version';
import * as tex_util from './tex_util';
import {TextureConfig} from './tex_util';
import * as webgl_util from './webgl_util';

export function createVertexShader(gl: WebGLRenderingContext): WebGLShader {
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
  return webgl_util.createVertexShader(gl, vertexShaderSource);
}

export function createVertexBuffer(gl: WebGLRenderingContext): WebGLBuffer {
  // [x y z u v] * [upper-left, lower-left, upper-right, lower-right]
  const vertexArray = new Float32Array(
      [-1, 1, 0, 0, 1, -1, -1, 0, 0, 0, 1, 1, 0, 1, 1, 1, -1, 0, 1, 0]);
  return webgl_util.createStaticVertexBuffer(gl, vertexArray);
}

export function createIndexBuffer(gl: WebGLRenderingContext): WebGLBuffer {
  // OpenGL (and WebGL) have "CCW == front" winding
  const triangleVertexIndices = new Uint16Array([0, 1, 2, 2, 1, 3]);
  return webgl_util.createStaticIndexBuffer(gl, triangleVertexIndices);
}

function createAndConfigureTexture(
    gl: WebGLRenderingContext, width: number, height: number,
    internalFormat: number, textureFormat: number,
    textureType: number): WebGLTexture {
  webgl_util.validateTextureSize(width, height);
  const texture = webgl_util.createTexture(gl);

  const tex2d = gl.TEXTURE_2D;
  webgl_util.callAndCheck(gl, () => gl.bindTexture(tex2d, texture));
  webgl_util.callAndCheck(
      gl, () => gl.texParameteri(tex2d, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE));
  webgl_util.callAndCheck(
      gl, () => gl.texParameteri(tex2d, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE));
  webgl_util.callAndCheck(
      gl, () => gl.texParameteri(tex2d, gl.TEXTURE_MIN_FILTER, gl.NEAREST));
  webgl_util.callAndCheck(
      gl, () => gl.texParameteri(tex2d, gl.TEXTURE_MAG_FILTER, gl.NEAREST));
  webgl_util.callAndCheck(
      gl,
      () => gl.texImage2D(
          tex2d, 0, internalFormat, width, height, 0, textureFormat,
          textureType, null));
  webgl_util.callAndCheck(gl, () => gl.bindTexture(gl.TEXTURE_2D, null));
  return texture;
}

export function getInternalFormatForFloat32MatrixTexture(
    textureConfig: TextureConfig) {
  return textureConfig.internalFormatFloat;
}

export function createFloat32MatrixTexture(
    gl: WebGLRenderingContext, rows: number, columns: number,
    textureConfig: TextureConfig): WebGLTexture {
  const [width, height] =
      tex_util.getUnpackedMatrixTextureShapeWidthHeight(rows, columns);
  return createAndConfigureTexture(
      gl, width, height,
      getInternalFormatForFloat32MatrixTexture(textureConfig),
      textureConfig.textureFormatFloat, gl.FLOAT);
}

export function getInternalFormatForFloat16MatrixTexture(
    textureConfig: TextureConfig) {
  return textureConfig.internalFormatHalfFloat;
}

export function createFloat16MatrixTexture(
    gl: WebGLRenderingContext, rows: number, columns: number,
    textureConfig: TextureConfig): WebGLTexture {
  const [width, height] =
      tex_util.getUnpackedMatrixTextureShapeWidthHeight(rows, columns);
  return createAndConfigureTexture(
      gl, width, height,
      getInternalFormatForFloat16MatrixTexture(textureConfig),
      textureConfig.textureFormatFloat, textureConfig.textureTypeHalfFloat);
}

export function getInternalFormatForUnsignedBytesMatrixTexture(
    textureConfig: TextureConfig) {
  return textureConfig.downloadTextureFormat;
}

export function createUnsignedBytesMatrixTexture(
    gl: WebGLRenderingContext, rows: number, columns: number,
    textureConfig: TextureConfig): WebGLTexture {
  const [width, height] =
      tex_util.getUnpackedMatrixTextureShapeWidthHeight(rows, columns);
  return createAndConfigureTexture(
      gl, width, height,
      getInternalFormatForUnsignedBytesMatrixTexture(textureConfig), gl.RGBA,
      gl.UNSIGNED_BYTE);
}

export function getInternalFormatForPackedMatrixTexture(
    textureConfig: TextureConfig) {
  return textureConfig.internalFormatPackedFloat;
}

export function createPackedMatrixTexture(
    gl: WebGLRenderingContext, rows: number, columns: number,
    textureConfig: TextureConfig): WebGLTexture {
  const [width, height] =
      tex_util.getPackedMatrixTextureShapeWidthHeight(rows, columns);
  return createAndConfigureTexture(
      gl, width, height, getInternalFormatForPackedMatrixTexture(textureConfig),
      gl.RGBA, gl.FLOAT);
}

export function getInternalFormatForFloat16PackedMatrixTexture(
    textureConfig: TextureConfig) {
  return textureConfig.internalFormatPackedHalfFloat;
}

export function createFloat16PackedMatrixTexture(
    gl: WebGLRenderingContext, rows: number, columns: number,
    textureConfig: TextureConfig): WebGLTexture {
  const [width, height] =
      tex_util.getPackedMatrixTextureShapeWidthHeight(rows, columns);
  return createAndConfigureTexture(
      gl, width, height,
      getInternalFormatForFloat16PackedMatrixTexture(textureConfig), gl.RGBA,
      textureConfig.textureTypeHalfFloat);
}

export function bindVertexProgramAttributeStreams(
    gl: WebGLRenderingContext, program: WebGLProgram,
    vertexBuffer: WebGLBuffer): boolean {
  const posOffset = 0;               // x is the first buffer element
  const uvOffset = 3 * 4;            // uv comes after [x y z]
  const stride = (3 * 4) + (2 * 4);  // xyz + uv, each entry is 4-byte float.
  webgl_util.callAndCheck(
      gl, () => gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer));
  const success = webgl_util.bindVertexBufferToProgramAttribute(
      gl, program, 'clipSpacePos', vertexBuffer, 3, stride, posOffset);
  return success &&
      webgl_util.bindVertexBufferToProgramAttribute(
          gl, program, 'uv', vertexBuffer, 2, stride, uvOffset);
}

export function uploadDenseMatrixToTexture(
    gl: WebGLRenderingContext, texture: WebGLTexture, width: number,
    height: number, data: TypedArray, textureConfig: TextureConfig) {
  webgl_util.callAndCheck(gl, () => gl.bindTexture(gl.TEXTURE_2D, texture));

  let dataForUpload: TypedArray, texelDataType: number, internalFormat: number;
  if (data instanceof Uint8Array) {
    dataForUpload = new Uint8Array(width * height * 4);
    texelDataType = gl.UNSIGNED_BYTE;
    internalFormat = gl.RGBA;
  } else {
    dataForUpload = new Float32Array(width * height * 4);
    texelDataType = gl.FLOAT;
    internalFormat = textureConfig.internalFormatPackedFloat;
  }

  dataForUpload.set(data);

  webgl_util.callAndCheck(
      gl,
      () => gl.texImage2D(
          gl.TEXTURE_2D, 0, internalFormat, width, height, 0, gl.RGBA,
          texelDataType, dataForUpload));

  webgl_util.callAndCheck(gl, () => gl.bindTexture(gl.TEXTURE_2D, null));
}

export function uploadPixelDataToTexture(
    gl: WebGLRenderingContext, texture: WebGLTexture,
    pixels: PixelData|ImageData|HTMLImageElement|HTMLCanvasElement|
    HTMLVideoElement|ImageBitmap) {
  webgl_util.callAndCheck(gl, () => gl.bindTexture(gl.TEXTURE_2D, texture));
  if ((pixels as PixelData).data instanceof Uint8Array) {
    webgl_util.callAndCheck(
        gl,
        () => gl.texImage2D(
            gl.TEXTURE_2D, 0, gl.RGBA, pixels.width, pixels.height, 0, gl.RGBA,
            gl.UNSIGNED_BYTE, (pixels as PixelData).data));
  } else {
    webgl_util.callAndCheck(
        gl,
        () => gl.texImage2D(
            gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE,
            pixels as ImageData | HTMLImageElement | HTMLCanvasElement |
                HTMLVideoElement|ImageBitmap));
  }

  webgl_util.callAndCheck(gl, () => gl.bindTexture(gl.TEXTURE_2D, null));
}

export function createBufferFromOutputTexture(
    gl2: WebGL2RenderingContext, rows: number, columns: number,
    textureConfig: TextureConfig): WebGLBuffer {
  // Create and bind the buffer.
  const buffer = gl2.createBuffer();
  webgl_util.callAndCheck(
      gl2, () => gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, buffer));

  // Initialize the buffer to the size of the texture in bytes.
  const bytesPerFloat = 4;
  const valuesPerTexel = 4;
  const bufferSizeBytes = bytesPerFloat * valuesPerTexel * rows * columns;

  webgl_util.callAndCheck(
      gl2,
      () => gl2.bufferData(
          gl2.PIXEL_PACK_BUFFER, bufferSizeBytes, gl2.STREAM_READ));

  // Enqueue a command on the GPU command queue to copy of texture into the
  // buffer.
  webgl_util.callAndCheck(
      gl2, () => gl2.readPixels(0, 0, columns, rows, gl2.RGBA, gl2.FLOAT, 0));

  webgl_util.callAndCheck(
      gl2, () => gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, null));

  return buffer;
}

export function downloadFloat32MatrixFromBuffer(
    gl: WebGLRenderingContext, buffer: WebGLBuffer,
    size: number): Float32Array {
  const gl2 = gl as WebGL2RenderingContext;

  const downloadTarget = new Float32Array(size);

  gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, buffer);
  gl2.getBufferSubData(gl2.PIXEL_PACK_BUFFER, 0, downloadTarget);
  gl2.bindBuffer(gl2.PIXEL_PACK_BUFFER, null);

  return downloadTarget;
}

export function downloadByteEncodedFloatMatrixFromOutputTexture(
    gl: WebGLRenderingContext, rows: number, columns: number,
    textureConfig: TextureConfig) {
  const [w, h] =
      tex_util.getUnpackedMatrixTextureShapeWidthHeight(rows, columns);

  const numChannels = 4;
  const downloadTarget = new Uint8Array(
      tex_util.getUnpackedArraySizeFromMatrixSize(rows * columns, numChannels));

  webgl_util.callAndCheck(
      gl,
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

  return downloadTarget;
}

export function downloadMatrixFromPackedOutputTexture(
    gl: WebGLRenderingContext, physicalRows: number,
    physicalCols: number): Float32Array {
  const packedRGBA = new Float32Array(physicalRows * physicalCols * 4);
  webgl_util.callAndCheck(
      gl,
      () => gl.readPixels(
          0, 0, physicalCols, physicalRows, gl.RGBA, gl.FLOAT, packedRGBA));

  return packedRGBA;
}
