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

import {env} from '@tensorflow/tfjs-core';

import {GPGPUContext} from './gpgpu_context';
import {getPackedMatrixTextureShapeWidthHeight, getUnpackedMatrixTextureShapeWidthHeight, PhysicalTextureType, TextureUsage} from './tex_util';

export class TextureManager {
  private numUsedTextures = 0;
  private numFreeTextures = 0;
  private numBytesAllocated = 0;
  private numBytesFree = 0;  // How many bytes that have been allocated
                             // are available for reuse.
  private freeTextures: {[shape: string]: WebGLTexture[]} = {};
  private logEnabled = false;
  private usedTextures: {[shape: string]: WebGLTexture[]} = {};

  constructor(private gpgpu: GPGPUContext) {}

  computeBytes(shape: [number, number], textureType: PhysicalTextureType):
      number {
    const [width, height] =
        getUnpackedMatrixTextureShapeWidthHeight(shape[0], shape[1]);
    const numElements = width * height;

    const [packedWidth, packedHeight] =
        getPackedMatrixTextureShapeWidthHeight(shape[0], shape[1]);
    const numPackedElements = packedWidth * packedHeight;

    if (textureType === PhysicalTextureType.PACKED_2X2_FLOAT32) {
      return numPackedElements *
          numBytesForInternalFormat(
                 this.gpgpu.gl,
                 this.gpgpu.textureConfig.internalFormatPackedFloat);
    } else if (textureType === PhysicalTextureType.PACKED_2X2_FLOAT16) {
      return numPackedElements *
          numBytesForInternalFormat(
                 this.gpgpu.gl,
                 this.gpgpu.textureConfig.internalFormatPackedHalfFloat);
    } else if (textureType === PhysicalTextureType.PACKED_4X1_UNSIGNED_BYTE) {
      return numElements *
          numBytesForInternalFormat(this.gpgpu.gl, this.gpgpu.gl.RGBA);
    } else if (textureType === PhysicalTextureType.UNPACKED_FLOAT16) {
      return numElements *
          numBytesForInternalFormat(
                 this.gpgpu.gl,
                 this.gpgpu.textureConfig.internalFormatHalfFloat);
    } else if (textureType === PhysicalTextureType.UNPACKED_FLOAT32) {
      return numElements *
          numBytesForInternalFormat(
                 this.gpgpu.gl, this.gpgpu.textureConfig.internalFormatFloat);
    }
    throw new Error(`Unknown physical texture type ${textureType}`);
  }

  acquireTexture(
      shapeRC: [number, number], usage: TextureUsage,
      isPacked: boolean): {texture: WebGLTexture, rows: number, cols: number} {
    const physicalTexType = getPhysicalFromLogicalTextureType(usage, isPacked);

    const shapeKey = getKeyFromTextureShape(shapeRC, physicalTexType, isPacked);
    if (!(shapeKey in this.freeTextures)) {
      this.freeTextures[shapeKey] = [];
    }
    if (!(shapeKey in this.usedTextures)) {
      this.usedTextures[shapeKey] = [];
    }

    if (this.freeTextures[shapeKey].length > 0) {
      this.numFreeTextures--;
      this.numUsedTextures++;
      this.numBytesFree -= this.computeBytes(shapeRC, physicalTexType);
      this.log();
      const newTexture = this.freeTextures[shapeKey].shift();
      this.usedTextures[shapeKey].push(newTexture);
      // return newTexture;
      const [rows, cols, ] = shapeKey.split('_');

      return {
        texture: newTexture,
        rows: parseInt(rows, 10),
        cols: parseInt(cols, 10)
      };
    }

    const freeTextures =
        Object.keys(this.freeTextures)
            .filter(key => this.freeTextures[key].length)
            .map(key => key.split('_'))
            .sort((a: string[], b: string[]) => {
              const aSize = parseInt(a[0], 10) * parseInt(a[1], 10);
              const bSize = parseInt(b[0], 10) * parseInt(b[1], 10);
              if (aSize < bSize) {
                return -1;
              }
              return 1;
            })
            .map(elements => elements.join('_'));

    let candidate;
    for (let i = 0; i < freeTextures.length; i++) {
      const key = freeTextures[i];
      const [rows, cols, type, candidateIsPacked] = key.split('_');
      const packedMatches =
          (candidateIsPacked === 'undefined' && isPacked == null) ||
          (candidateIsPacked === 'true' && isPacked);
      if (parseInt(type, 10) === physicalTexType && packedMatches) {
        if (parseInt(rows, 10) >= shapeRC[0] &&
            parseInt(cols, 10) >= shapeRC[1]) {
          candidate = key;
          break;
        }
      }
    }

    if (candidate) {
      const [rowsS, colsS, ] = candidate.split('_');
      const rows = parseInt(rowsS, 10);
      const cols = parseInt(colsS, 10);

      if ((shapeRC[0] * shapeRC[1]) / (rows * cols) > 0) {
        console.log('RECYCLING!!!!!');
        this.numFreeTextures--;
        this.numUsedTextures++;
        this.numBytesFree -= this.computeBytes([rows, cols], physicalTexType);
        this.log();
        const newTexture = this.freeTextures[candidate].shift();
        this.usedTextures[candidate].push(newTexture);
        return {texture: newTexture, rows, cols};
      }
    }
    this.numUsedTextures++;
    this.numBytesAllocated += this.computeBytes(shapeRC, physicalTexType);
    this.log();

    let newTexture: WebGLTexture;
    if (physicalTexType === PhysicalTextureType.PACKED_2X2_FLOAT32) {
      newTexture = this.gpgpu.createPackedMatrixTexture(shapeRC[0], shapeRC[1]);
    } else if (physicalTexType === PhysicalTextureType.PACKED_2X2_FLOAT16) {
      newTexture =
          this.gpgpu.createFloat16PackedMatrixTexture(shapeRC[0], shapeRC[1]);
    } else if (physicalTexType === PhysicalTextureType.UNPACKED_FLOAT32) {
      newTexture =
          this.gpgpu.createFloat32MatrixTexture(shapeRC[0], shapeRC[1]);
    } else if (physicalTexType === PhysicalTextureType.UNPACKED_FLOAT16) {
      newTexture =
          this.gpgpu.createFloat16MatrixTexture(shapeRC[0], shapeRC[1]);

    } else if (
        physicalTexType === PhysicalTextureType.PACKED_4X1_UNSIGNED_BYTE) {
      newTexture =
          this.gpgpu.createUnsignedBytesMatrixTexture(shapeRC[0], shapeRC[1]);
    }
    this.usedTextures[shapeKey].push(newTexture);

    // return newTexture;
    const [rows, cols, ] = shapeKey.split('_');
    return {
      texture: newTexture,
      rows: parseInt(rows, 10),
      cols: parseInt(cols, 10)
    };
  }

  releaseTexture(
      texture: WebGLTexture, shape: [number, number],
      logicalTexType: TextureUsage, isPacked: boolean, actualRows: number,
      actualCols: number): void {
    // console.log('rELEASE', actualRows, actualCols);
    if (this.freeTextures == null) {
      // Already disposed.
      return;
    }
    shape = [actualRows, actualCols];
    const physicalTexType =
        getPhysicalFromLogicalTextureType(logicalTexType, isPacked);
    const shapeKey = getKeyFromTextureShape(shape, physicalTexType, isPacked);
    // console.log('release tex', shapeKey);
    if (!(shapeKey in this.freeTextures)) {
      this.freeTextures[shapeKey] = [];
    }
    this.freeTextures[shapeKey].push(texture);
    this.numFreeTextures++;
    this.numUsedTextures--;
    this.numBytesFree += this.computeBytes(shape, physicalTexType);
    const texList = this.usedTextures[shapeKey];
    const texIndex = texList.indexOf(texture);
    if (texIndex < 0) {
      throw new Error(
          'Cannot release a texture that was never provided by this ' +
          'texture manager');
    }
    texList.splice(texIndex, 1);
    this.log();
  }

  private log() {
    if (!this.logEnabled) {
      return;
    }
    const total = this.numFreeTextures + this.numUsedTextures;
    console.log(
        'Free/Used', `${this.numFreeTextures} / ${this.numUsedTextures}`,
        `(${total})`);
    const freeRatio = this.numBytesFree / this.numBytesAllocated;
    console.log(`Bytes allocated: ${this.numBytesAllocated}`);
    console.log(
        `Bytes unused: ${this.numBytesFree} (${Math.round(100 * freeRatio)}%)`);
  }

  getNumBytesAllocated(): number {
    return this.numBytesAllocated;
  }

  getNumBytesFree(): number {
    return this.numBytesFree;
  }

  getNumUsedTextures(): number {
    return this.numUsedTextures;
  }

  getNumFreeTextures(): number {
    return this.numFreeTextures;
  }

  dispose() {
    if (this.freeTextures == null) {
      // Already disposed.
      return;
    }
    for (const texShape in this.freeTextures) {
      this.freeTextures[texShape].forEach(tex => {
        this.gpgpu.deleteMatrixTexture(tex);
      });
    }
    for (const texShape in this.usedTextures) {
      this.usedTextures[texShape].forEach(tex => {
        this.gpgpu.deleteMatrixTexture(tex);
      });
    }
    this.freeTextures = null;
    this.usedTextures = null;
    this.numUsedTextures = 0;
    this.numFreeTextures = 0;
    this.numBytesAllocated = 0;
    this.numBytesFree = 0;
  }
}

function numBytesForInternalFormat(
    gl: WebGLRenderingContext, internalFormat: number): number {
  if (internalFormat === (gl as any).R32F) {
    return 4;
  } else if (internalFormat === (gl as any).R16F) {
    return 2;
  } else if (internalFormat === (gl as any).RGBA32F) {
    return 16;
  } else if (internalFormat === gl.RGBA) {
    return 16;
  }
  throw new Error(`Unknown internal format ${internalFormat}`);
}

function getPhysicalTextureForRendering(isPacked: boolean):
    PhysicalTextureType {
  if (env().getBool('WEBGL_RENDER_FLOAT32_ENABLED')) {
    if (isPacked) {
      return PhysicalTextureType.PACKED_2X2_FLOAT32;
    }
    return PhysicalTextureType.UNPACKED_FLOAT32;
  }

  if (isPacked) {
    return PhysicalTextureType.PACKED_2X2_FLOAT16;
  }
  return PhysicalTextureType.UNPACKED_FLOAT16;
}

function getPhysicalFromLogicalTextureType(
    logicalTexType: TextureUsage, isPacked: boolean): PhysicalTextureType {
  if (logicalTexType === TextureUsage.UPLOAD) {
    return PhysicalTextureType.PACKED_2X2_FLOAT32;
  } else if (logicalTexType === TextureUsage.RENDER || logicalTexType == null) {
    return getPhysicalTextureForRendering(isPacked);
  } else if (
      logicalTexType === TextureUsage.DOWNLOAD ||
      logicalTexType === TextureUsage.PIXELS) {
    return PhysicalTextureType.PACKED_4X1_UNSIGNED_BYTE;
  }
  throw new Error(`Unknown logical texture type ${logicalTexType}`);
}

function getKeyFromTextureShape(
    shapeRowsCol: [number, number], physicalTexType: PhysicalTextureType,
    isPacked: boolean): string {
  return `${shapeRowsCol[0]}_${shapeRowsCol[1]}_${physicalTexType}_${isPacked}`;
}
