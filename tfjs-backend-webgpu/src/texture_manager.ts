/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

export function GetFormatSize(texFormat: GPUTextureFormat) {
  switch (texFormat) {
    case 'r32float':
      return 4;
    case 'rgba8unorm':
      return 4;
    default:
      break;
  }
  return 4;
}

export class TextureManager {
  private numUsedTextures = 0;
  private numFreeTextures = 0;
  private freeTextures: Map<string, GPUTexture[]> = new Map();
  private usedTextures: Map<string, GPUTexture[]> = new Map();

  public numBytesUsed = 0;
  public numBytesAllocated = 0;

  constructor(private device: GPUDevice) {}

  acquireTexture(
      byteSize: number, texFormat: GPUTextureFormat,
      usages: GPUTextureUsageFlags) {
    const key = getTextureKey(byteSize, texFormat, usages);
    if (!this.freeTextures.has(key)) {
      this.freeTextures.set(key, []);
    }

    if (!this.usedTextures.has(key)) {
      this.usedTextures.set(key, []);
    }

    this.numBytesUsed += byteSize;
    this.numUsedTextures++;

    if (this.freeTextures.get(key).length > 0) {
      this.numFreeTextures--;

      const newTexture = this.freeTextures.get(key).shift();
      this.usedTextures.get(key).push(newTexture);
      return newTexture;
    }
    const formatSize = GetFormatSize(texFormat);
    this.numBytesAllocated += byteSize;
    const newTexture = this.device.createTexture({
      size: {width: byteSize / formatSize, height: 1, depth: 1},
      format: texFormat,
      dimension: '2d',
      usage: usages,
    });
    this.usedTextures.get(key).push(newTexture);

    return newTexture;
  }

  releaseTexture(
      texture: GPUTexture, byteSize: number, texFormat: GPUTextureFormat,
      usage: GPUTextureUsageFlags) {
    if (this.freeTextures == null) {
      return;
    }

    const key = getTextureKey(byteSize, texFormat, usage);
    if (!this.freeTextures.has(key)) {
      this.freeTextures.set(key, []);
    }

    this.freeTextures.get(key).push(texture);
    this.numFreeTextures++;
    this.numUsedTextures--;

    const textureList = this.usedTextures.get(key);
    const textureIndex = textureList.indexOf(texture);
    if (textureIndex < 0) {
      throw new Error(
          'Cannot release a Texture that was never provided by this ' +
          'Texture manager');
    }
    textureList.splice(textureIndex, 1);
    this.numBytesUsed -= byteSize;
  }

  getNumUsedTextures(): number {
    return this.numUsedTextures;
  }

  getNumFreeTextures(): number {
    return this.numFreeTextures;
  }

  reset() {
    this.freeTextures = new Map();
    this.usedTextures = new Map();
    this.numUsedTextures = 0;
    this.numFreeTextures = 0;
    this.numBytesUsed = 0;
    this.numBytesAllocated = 0;
  }

  dispose() {
    if (this.freeTextures == null && this.usedTextures == null) {
      return;
    }

    this.freeTextures.forEach((textures) => {
      textures.forEach(tex => {
        tex.destroy();
      });
    });

    this.usedTextures.forEach((textures) => {
      textures.forEach(tex => {
        tex.destroy();
      });
    });

    this.freeTextures = null;
    this.usedTextures = null;
    this.numUsedTextures = 0;
    this.numFreeTextures = 0;
    this.numBytesUsed = 0;
    this.numBytesAllocated = 0;
  }
}

function getTextureKey(
    byteSize: number, format: GPUTextureFormat, usage: GPUTextureUsageFlags) {
  return `${byteSize}_${format}_${usage}`;
}
