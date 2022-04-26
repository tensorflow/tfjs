/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

export class TextureManager {
  private numUsedTextures = 0;
  private numFreeTextures = 0;
  private freeTextures: Map<string, GPUTexture[]> = new Map();
  private usedTextures: Map<string, GPUTexture[]> = new Map();

  public numBytesUsed = 0;
  public numBytesAllocated = 0;

  constructor(private device: GPUDevice) {}

  acquireTexture(
      width: number, height: number, format: GPUTextureFormat,
      usage: GPUTextureUsageFlags) {
    const bytesPerElement = getBytesPerElement(format);
    const byteSize = width * height * bytesPerElement;
    const key = getTextureKey(width, height, format, usage);
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

    this.numBytesAllocated += byteSize;

    const newTexture = this.device.createTexture({
      size: [width, height],
      format,
      usage,
    });
    this.usedTextures.get(key).push(newTexture);

    return newTexture;
  }

  releaseTexture(
      texture: GPUTexture, width: number, height: number,
      format: GPUTextureFormat, usage: GPUTextureUsageFlags) {
    if (this.freeTextures.size === 0) {
      return;
    }

    const key = getTextureKey(width, height, format, usage);
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
          'Cannot release a texture that was never provided by this ' +
          'texture manager');
    }
    textureList.splice(textureIndex, 1);
    const bytesPerElement = getBytesPerElement(format);
    const byteSize = width * height * bytesPerElement;
    this.numBytesUsed -= byteSize;
  }

  getNumUsedTextures(): number {
    return this.numUsedTextures;
  }

  getNumFreeTextures(): number {
    return this.numFreeTextures;
  }

  dispose() {
    this.freeTextures.forEach((textures, key) => {
      textures.forEach(texture => {
        texture.destroy();
      });
    });

    this.usedTextures.forEach((textures, key) => {
      textures.forEach(texture => {
        texture.destroy();
      });
    });

    this.freeTextures = new Map();
    this.usedTextures = new Map();
    this.numUsedTextures = 0;
    this.numFreeTextures = 0;
    this.numBytesUsed = 0;
    this.numBytesAllocated = 0;
  }
}

function getTextureKey(
    width: number, height: number, format: GPUTextureFormat,
    usage: GPUTextureUsageFlags) {
  return `${width}_${height}_${format}_${usage}`;
}

function getBytesPerElement(format: GPUTextureFormat) {
  if (format === 'rgba8unorm') {
    return 16;
  } else {
    throw new Error(`${format} is not supported!`);
  }
}
