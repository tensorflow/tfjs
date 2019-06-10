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

export function getMatrixSizeFromUnpackedArraySize(
    unpackedSize: number, channelsPerTexture: number): number {
  if (unpackedSize % channelsPerTexture !== 0) {
    throw new Error(
        `unpackedSize (${unpackedSize}) must be a multiple of ` +
        `${channelsPerTexture}`);
  }
  return unpackedSize / channelsPerTexture;
}

export function decodeMatrixFromUnpackedArray(
    unpackedArray: Float32Array, matrix: Float32Array,
    channelsPerTexture: number) {
  const requiredSize = getMatrixSizeFromUnpackedArraySize(
      unpackedArray.length, channelsPerTexture);
  if (matrix.length < requiredSize) {
    throw new Error(
        `matrix length (${matrix.length}) must be >= ${requiredSize}`);
  }
  let dst = 0;
  for (let src = 0; src < unpackedArray.length; src += channelsPerTexture) {
    matrix[dst++] = unpackedArray[src];
  }
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

export function decodeMatrixFromPackedRGBA(
    packedRGBA: Float32Array, batches: number, rows: number, columns: number,
    matrix: Float32Array): Float32Array {
  const requiredSize = rows * columns;
  if (matrix.length < requiredSize) {
    throw new Error(
        `matrix length (${matrix.length}) must be >= ${requiredSize}`);
  }

  const oddWidth = (columns % 2) === 1;
  const oddHeight = (rows % 2) === 1;
  const widthInFullBlocks = Math.floor(columns / 2);
  const heightInFullBlocks = Math.floor(rows / 2);

  const texelsPerRow = Math.ceil(columns / 2);
  const texelsPerBatch = texelsPerRow * Math.ceil(rows / 2);

  const flattenedMatrixSize =
      util.nearestLargerEven(rows) * util.nearestLargerEven(columns);

  for (let batch = 0; batch < batches; batch++) {
    const batchOffset = batch * rows * columns;
    const sourceOffset = batch * flattenedMatrixSize;

    // loop over full 2x2 blocks
    {
      const srcStride = oddWidth ? 4 : 0;
      const dstStride = columns + (oddWidth ? 1 : 0);
      let src = sourceOffset;
      let dstRow1 = batchOffset;
      let dstRow2 = batchOffset + columns;
      for (let blockY = 0; blockY < heightInFullBlocks; ++blockY) {
        for (let blockX = 0; blockX < widthInFullBlocks; ++blockX) {
          matrix[dstRow1++] = packedRGBA[src++];
          matrix[dstRow1++] = packedRGBA[src++];
          matrix[dstRow2++] = packedRGBA[src++];
          matrix[dstRow2++] = packedRGBA[src++];
        }
        src += srcStride;
        dstRow1 += dstStride;
        dstRow2 += dstStride;
      }
    }

    // loop down final column
    if (oddWidth) {
      let src = sourceOffset + (texelsPerRow - 1) * 4;
      let dst = batchOffset + columns - 1;
      const srcStride = texelsPerRow * 4;
      const dstStride = 2 * columns;
      for (let blockY = 0; blockY < heightInFullBlocks; ++blockY) {
        matrix[dst] = packedRGBA[src];
        matrix[dst + columns] = packedRGBA[src + 2];
        src += srcStride;
        dst += dstStride;
      }
    }

    // loop across final row
    if (oddHeight) {
      let src = sourceOffset + (texelsPerBatch - texelsPerRow) * 4;
      let dst = batchOffset + (rows - 1) * columns;
      for (let blockX = 0; blockX < widthInFullBlocks; ++blockX) {
        matrix[dst++] = packedRGBA[src++];
        matrix[dst++] = packedRGBA[src++];
        src += 2;
      }

      // fill in bottom-right cell
      if (oddWidth) {
        matrix[batchOffset + (rows * columns) - 1] = packedRGBA[src];
      }
    }
  }

  return matrix;
}
