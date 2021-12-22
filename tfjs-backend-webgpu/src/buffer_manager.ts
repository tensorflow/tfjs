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
  public numBytesAllocated = 0;

  constructor(private device: GPUDevice) {}

  acquireUploadBuffer(byteSize: number, usage: GPUBufferUsageFlags) {
    return this.acquireBuffer(byteSize, usage, true);
  }

  acquireBuffer(
      byteSize: number, usage: GPUBufferUsageFlags, mappedAtCreation = false) {
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

    this.numBytesAllocated += byteSize;
    const newBuffer =
        this.device.createBuffer({mappedAtCreation, size: byteSize, usage});
    this.usedBuffers.get(key).push(newBuffer);

    return newBuffer;
  }

  releaseBuffer(
      buffer: GPUBuffer, byteSize: number, usage: GPUBufferUsageFlags) {
    if (this.freeBuffers.size === 0) {
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

  releaseUploadBuffer(
      buffer: GPUBuffer, byteSize: number, usage: GPUBufferUsageFlags) {
    buffer.mapAsync(GPUMapMode.WRITE)
        .then(
            () => {
              this.releaseBuffer(buffer, byteSize, usage);
            },
            (err) => {
                // Do nothing;
            });
  }

  getNumUsedBuffers(): number {
    return this.numUsedBuffers;
  }

  getNumFreeBuffers(): number {
    return this.numFreeBuffers;
  }

  dispose() {
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

    this.freeBuffers = new Map();
    this.usedBuffers = new Map();
    this.numUsedBuffers = 0;
    this.numFreeBuffers = 0;
    this.numBytesUsed = 0;
    this.numBytesAllocated = 0;
  }
}

function getBufferKey(byteSize: number, usage: GPUBufferUsageFlags) {
  return `${byteSize}_${usage}`;
}
