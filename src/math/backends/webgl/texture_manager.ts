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

import {GPGPUContext} from './gpgpu_context';

export class TextureManager {
  private numUsedTextures = 0;
  private numFreeTextures = 0;
  private freeTextures: {[shape: string]: WebGLTexture[]} = {};
  private logEnabled = false;
  private allocatedTextures: WebGLTexture[] = [];
  private usedTextureCount: {[shape: string]: number} = {};

  constructor(private gpgpu: GPGPUContext) {}

  acquireTexture(shapeRC: [number, number]): WebGLTexture {
    const shapeKey = getKeyFromTextureShape(shapeRC);
    if (!(shapeKey in this.freeTextures)) {
      this.freeTextures[shapeKey] = [];
    }
    if (!(shapeKey in this.usedTextureCount)) {
      this.usedTextureCount[shapeKey] = 0;
    }
    this.usedTextureCount[shapeKey]++;

    if (this.freeTextures[shapeKey].length > 0) {
      this.numFreeTextures--;
      this.numUsedTextures++;
      this.log();
      return this.freeTextures[shapeKey].shift();
    }
    this.numUsedTextures++;
    this.log();

    const newTexture = this.gpgpu.createMatrixTexture(shapeRC[0], shapeRC[1]);
    this.allocatedTextures.push(newTexture);
    return newTexture;
  }

  releaseTexture(texture: WebGLTexture, shape: [number, number]): void {
    const shapeKey = getKeyFromTextureShape(shape);
    if (!(shapeKey in this.freeTextures)) {
      this.freeTextures[shapeKey] = [];
    }
    this.freeTextures[shapeKey].push(texture);
    this.numFreeTextures++;
    this.numUsedTextures--;
    this.usedTextureCount[shapeKey]--;
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
  }

  getNumUsedTextures(): number {
    return this.numUsedTextures;
  }

  getNumFreeTextures(): number {
    return this.numFreeTextures;
  }

  dispose() {
    if (this.allocatedTextures == null) {
      // Already disposed.
      return;
    }
    this.allocatedTextures.forEach(texture => {
      this.gpgpu.deleteMatrixTexture(texture);
    });
    this.freeTextures = null;
    this.allocatedTextures = null;
    this.usedTextureCount = null;
    this.numUsedTextures = 0;
    this.numFreeTextures = 0;
  }
}

function getKeyFromTextureShape(shapeRowsCol: [number, number]): string {
  return `${shapeRowsCol[0]}_${shapeRowsCol[1]}`;
}
