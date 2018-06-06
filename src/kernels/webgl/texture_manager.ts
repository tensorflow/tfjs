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
import {TextureType} from './tex_util';

export class TextureManager {
  private numUsedTextures = 0;
  private numFreeTextures = 0;
  private freeTextures: {[shape: string]: WebGLTexture[]} = {};
  private logEnabled = false;
  private usedTextures: {[shape: string]: WebGLTexture[]} = {};

  constructor(private gpgpu: GPGPUContext) {}

  acquireTexture(shapeRC: [number, number], texType = TextureType.FLOAT):
      WebGLTexture {
    const shapeKey = getKeyFromTextureShape(shapeRC, texType);
    if (!(shapeKey in this.freeTextures)) {
      this.freeTextures[shapeKey] = [];
    }
    if (!(shapeKey in this.usedTextures)) {
      this.usedTextures[shapeKey] = [];
    }

    if (this.freeTextures[shapeKey].length > 0) {
      this.numFreeTextures--;
      this.numUsedTextures++;
      this.log();
      const newTexture = this.freeTextures[shapeKey].shift();
      this.usedTextures[shapeKey].push(newTexture);
      return newTexture;
    }
    this.numUsedTextures++;
    this.log();

    const newTexture = this.gpgpu.createMatrixTexture(shapeRC[0], shapeRC[1]);
    this.usedTextures[shapeKey].push(newTexture);
    return newTexture;
  }

  releaseTexture(
      texture: WebGLTexture, shape: [number, number],
      texType = TextureType.FLOAT): void {
    const shapeKey = getKeyFromTextureShape(shape, texType);
    if (!(shapeKey in this.freeTextures)) {
      this.freeTextures[shapeKey] = [];
    }
    this.freeTextures[shapeKey].push(texture);
    this.numFreeTextures++;
    this.numUsedTextures--;
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
  }
}

function getKeyFromTextureShape(
    shapeRowsCol: [number, number], texType: TextureType): string {
  return `${shapeRowsCol[0]}_${shapeRowsCol[1]}_${texType}`;
}
