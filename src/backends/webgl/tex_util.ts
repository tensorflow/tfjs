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

import {DataId, Tensor} from '../../tensor';
import {DataType, DataValues} from '../../types';
import * as util from '../../util';

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
  values?: DataValues;
  texture?: WebGLTexture;
  // For complex numbers, the real and imaginary parts are stored as their own
  // individual tensors, with a parent joining the two with the
  // complexTensors field. When this is defined, texture will be null.
  complexTensors?: {real: Tensor, imag: Tensor};
  /** [rows, columns] shape of the texture. */
  texShape?: [number, number];
  usage?: TextureUsage;
  isPacked?: boolean;

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
