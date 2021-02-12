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

import {backend_util, DataId, DataType, env, TensorInfo, util} from '@tensorflow/tfjs-core';

export enum PackingScheme {
  /**
   * All values in a single texel are densely packed without any constraints.
   *
   * This is how the shader encodes a tensor with shape = [2, 3, 4]
   * (indices are [batch, row, col]).
   *
   * 000|001   010|011   020|021
   * -------   -------   -------
   * 002|003   012|013   022|023
   *
   * 100|101   110|111   120|121
   * -------   -------   -------
   * 102|103   112|113   122|123
   *
   */
  DENSE,

  /**
   * Single texels contain only values from the same batch, and from adjacent
   * rows and columns.
   *
   * This is how the shader encodes a tensor with shape = [2, 3, 5]
   * (indices are [batch, row, col]).
   *
   * 000|001   002|003   004|xxx   020|021   022|023   024|xxx
   * -------   -------   -------   -------   -------   -------
   * 010|011   012|013   014|xxx   xxx|xxx   xxx|xxx   xxx|xxx
   *
   * 100|101   102|103   104|xxx   120|121   122|123   124|xxx
   * -------   -------   -------   -------   -------   -------
   * 110|111   112|113   114|xxx   xxx|xxx   xxx|xxx   xxx|xxx
   *
   */
  SHARED_BATCH
}

export enum TextureUsage {
  RENDER,
  UPLOAD,
  PIXELS,
  DOWNLOAD
}

export enum PhysicalTextureType {
  UNPACKED_FLOAT16,
  UNPACKED_FLOAT32,
  PACKED_4X1_UNSIGNED_BYTE,
  PACKED_2X2_FLOAT32,
  PACKED_2X2_FLOAT16
}

export interface TextureData {
  // Required.
  shape: number[];
  dtype: DataType;

  // Optional.
  values?: backend_util.BackendValues;
  texture?: WebGLTexture;
  // For complex numbers, the real and imaginary parts are stored as their own
  // individual tensorInfos, with a parent joining the two with the
  // complexTensors field. When this is defined, texture will be null.
  complexTensorInfos?: {real: TensorInfo, imag: TensorInfo};
  /** [rows, columns] shape of the texture. */
  texShape?: [number, number];
  usage?: TextureUsage;
  isPacked?: boolean;

  refCount: number;

  // Available when the tensor has been sliced.
  slice?: {
    // Offset in the 'flat index' space.
    flatOffset: number;
    // Used for counting how many sliced tensors point to the same texture.
    origDataId: DataId;
  };
}

export function getUnpackedMatrixTextureShapeWidthHeight(
    rows: number, columns: number): [number, number] {
  return [columns, rows];
}

export function getUnpackedArraySizeFromMatrixSize(
    matrixSize: number, channelsPerTexture: number): number {
  return matrixSize * channelsPerTexture;
}

export function getColorMatrixTextureShapeWidthHeight(
    rows: number, columns: number): [number, number] {
  return [columns * 4, rows];
}

/**
 * Get shape for densely packed RGBA texture.
 */
export function getDenseTexShape(shape: number[]): [number, number] {
  const size = util.sizeFromShape(shape);
  const texelsNeeded = Math.ceil(size / 4);
  return util.sizeToSquarishShape(texelsNeeded);
}

export function getMatrixSizeFromUnpackedArraySize(
    unpackedSize: number, channelsPerTexture: number): number {
  if (unpackedSize % channelsPerTexture !== 0) {
    throw new Error(
        `unpackedSize (${unpackedSize}) must be a multiple of ` +
        `${channelsPerTexture}`);
  }
  return unpackedSize / channelsPerTexture;
}

export function decodeMatrixFromUnpackedColorRGBAArray(
    unpackedArray: Float32Array, matrix: Float32Array, channels: number) {
  const requiredSize = unpackedArray.length * channels / 4;
  if (matrix.length < requiredSize) {
    throw new Error(
        `matrix length (${matrix.length}) must be >= ${requiredSize}`);
  }
  let dst = 0;
  for (let src = 0; src < unpackedArray.length; src += 4) {
    for (let c = 0; c < channels; c++) {
      matrix[dst++] = unpackedArray[src + c];
    }
  }
}

export function getPackedMatrixTextureShapeWidthHeight(
    rows: number, columns: number): [number, number] {
  return [
    Math.max(1, Math.ceil(columns / 2)), Math.max(1, Math.ceil(rows / 2))
  ];
}

export function getPackedRGBAArraySizeFromMatrixShape(
    rows: number, columns: number): number {
  const [w, h] = getPackedMatrixTextureShapeWidthHeight(rows, columns);
  return w * h * 4;
}

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
  textureTypeFloat: number;
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
  let textureTypeFloat: number;

  if (env().getNumber('WEBGL_VERSION') === 2) {
    internalFormatFloat = glany.R32F;
    internalFormatHalfFloat = glany.R16F;
    internalFormatPackedHalfFloat = glany.RGBA16F;
    internalFormatPackedFloat = glany.RGBA32F;
    textureFormatFloat = glany.RED;
    downloadUnpackNumChannels = 4;
    defaultNumChannels = 1;
    textureTypeHalfFloat = glany.HALF_FLOAT;
    textureTypeFloat = glany.FLOAT;
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
    textureTypeFloat = gl.FLOAT;
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
    textureTypeHalfFloat,
    textureTypeFloat
  };
}
