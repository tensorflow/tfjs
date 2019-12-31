/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

export class BufferManager {
  private numUsedBuffers = 0;
  private numFreeBuffers = 0;
  private freeBuffers: Map<string, GPUBuffer[]> = new Map();
  private usedBuffers: Map<string, GPUBuffer[]> = new Map();

  public numBytesUsed = 0;

  constructor(private device: GPUDevice) {}

  acquireBuffer(byteSize: number, usage: GPUBufferUsage) {
    const key = getBufferKey(byteSize, usage);
    if (!this.freeBuffers.has(key)) {
      this.freeBuffers.set(key, []);
    }

    if (!this.usedBuffers.has(key)) {
      this.usedBuffers.set(key, []);
    }

    this.numBytesUsed += byteSize;
    this.numUsedBuffers++;

    if (this.freeBuffers.get(key).length > 0) {
      this.numFreeBuffers--;

      const newBuffer = this.freeBuffers.get(key).shift();
      this.usedBuffers.get(key).push(newBuffer);
      return newBuffer;
    }

    const newBuffer = this.device.createBuffer({size: byteSize, usage});
    this.usedBuffers.get(key).push(newBuffer);

    return newBuffer;
  }

  releaseBuffer(buffer: GPUBuffer, byteSize: number, usage: GPUBufferUsage) {
    if (this.freeBuffers == null) {
      return;
    }

    const key = getBufferKey(byteSize, usage);
    if (!this.freeBuffers.has(key)) {
      this.freeBuffers.set(key, []);
    }

    this.freeBuffers.get(key).push(buffer);
    this.numFreeBuffers++;
    this.numUsedBuffers--;

    const bufferList = this.usedBuffers.get(key);
    const bufferIndex = bufferList.indexOf(buffer);
    if (bufferIndex < 0) {
      throw new Error(
          'Cannot release a buffer that was never provided by this ' +
          'buffer manager');
    }
    bufferList.splice(bufferIndex, 1);
    this.numBytesUsed -= byteSize;
  }

  getNumUsedBuffers(): number {
    return this.numUsedBuffers;
  }

  getNumFreeBuffers(): number {
    return this.numFreeBuffers;
  }

  reset() {
    this.freeBuffers = new Map();
    this.usedBuffers = new Map();
    this.numUsedBuffers = 0;
    this.numFreeBuffers = 0;
  }

  dispose() {
    if (this.freeBuffers == null && this.usedBuffers == null) {
      return;
    }

    this.freeBuffers.forEach((buffers, key) => {
      buffers.forEach(buff => {
        buff.destroy();
      });
    });

    this.usedBuffers.forEach((buffers, key) => {
      buffers.forEach(buff => {
        buff.destroy();
      });
    });

    this.freeBuffers = null;
    this.usedBuffers = null;
    this.numUsedBuffers = 0;
    this.numFreeBuffers = 0;
  }
}

function getBufferKey(byteSize: number, usage: GPUBufferUsage) {
  return `${byteSize}_${usage}`;
}
