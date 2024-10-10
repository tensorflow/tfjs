/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
import {TypedArray} from '../types';
import * as util from '../util';

type BufferShard = {
  start: number,
  end: number,
  buffer: ArrayBuffer,
};

/**
 * Wraps a list of ArrayBuffers into a `slice()`-able object without allocating
 * a large ArrayBuffer.
 *
 * Allocating large ArrayBuffers (~2GB) can be unstable on Chrome. TFJS loads
 * its weights as a list of (usually) 4MB ArrayBuffers and then slices the
 * weight tensors out of them. For small models, it's safe to concatenate all
 * the weight buffers into a single ArrayBuffer and then slice the weight
 * tensors out of it, but for large models, a different approach is needed.
 */

export class CompositeArrayBuffer {
  private shards: BufferShard[] = [];
  private previousShardIndex = 0;
  private bufferUniformSize?: number;
  public readonly byteLength: number;

  /**
   * Concatenate a number of ArrayBuffers into one.
   *
   * @param buffers An array of ArrayBuffers to concatenate, or a single
   *     ArrayBuffer.
   * @returns Result of concatenating `buffers` in order.
   */
  static join(buffers?: ArrayBuffer[] | ArrayBuffer) {
    return new CompositeArrayBuffer(buffers).slice();
  }

  constructor(buffers?: ArrayBuffer | ArrayBuffer[] | TypedArray |
    TypedArray[]) {
    if (buffers == null) {
      return;
    }
    // Normalize the `buffers` input to be `ArrayBuffer[]`.
    if (!(buffers instanceof Array)) {
      buffers = [buffers];
    }
    buffers = buffers.map((bufferOrTypedArray) => {
      if (util.isTypedArray(bufferOrTypedArray)) {
        return bufferOrTypedArray.buffer;
      }
      return bufferOrTypedArray;
    });

    // Skip setting up shards if there are no buffers.
    if (buffers.length === 0) {
      return;
    }

    this.bufferUniformSize = buffers[0].byteLength;
    let start = 0;

    for (let i = 0; i < buffers.length; i++) {
      const buffer = buffers[i];
      // Check that all buffers except the last one have the same length.
      if (i !== buffers.length - 1 &&
        buffer.byteLength !== this.bufferUniformSize) {
        // Unset the buffer uniform size, since the buffer sizes are not
        // uniform.
        this.bufferUniformSize = undefined;
      }

      // Create the shards, including their start and end points.
      const end = start + buffer.byteLength;
      this.shards.push({ buffer, start, end });
      start = end;
    }

    // Set the byteLength
    if (this.shards.length === 0) {
      this.byteLength = 0;
    }
    this.byteLength = this.shards[this.shards.length - 1].end;
  }

  slice(start = 0, end = this.byteLength): ArrayBuffer {
    // If there are no shards, then the CompositeArrayBuffer was initialized
    // with no data.
    if (this.shards.length === 0) {
      return new ArrayBuffer(0);
    }

    // NaN is treated as zero for slicing. This matches ArrayBuffer's behavior.
    start = isNaN(Number(start)) ? 0 : start;
    end = isNaN(Number(end)) ? 0 : end;

    // Fix the bounds to within the array.
    start = Math.max(0, start);
    end = Math.min(this.byteLength, end);
    if (end <= start) {
      return new ArrayBuffer(0);
    }

    const startShardIndex = this.findShardForByte(start);
    if (startShardIndex === -1) {
      // This should not happen since the start and end indices are always
      // within 0 and the composite array's length.
      throw new Error(`Could not find start shard for byte ${start}`);
    }

    const size = end - start;
    const outputBuffer = new ArrayBuffer(size);
    const outputArray = new Uint8Array(outputBuffer);
    let sliced = 0;
    for (let i = startShardIndex; i < this.shards.length; i++) {
      const shard = this.shards[i];

      const globalStart = start + sliced;
      const localStart = globalStart - shard.start;
      const outputStart = sliced;

      const globalEnd = Math.min(end, shard.end);
      const localEnd = globalEnd - shard.start;

      const outputSlice = new Uint8Array(shard.buffer, localStart,
                                         localEnd - localStart);
      outputArray.set(outputSlice, outputStart);
      sliced += outputSlice.length;

      if (end < shard.end) {
        break;
      }
    }
    return outputBuffer;
  }

  /**
   * Get the index of the shard that contains the byte at `byteIndex`.
   */
  private findShardForByte(byteIndex: number): number {
    if (this.shards.length === 0 || byteIndex < 0 ||
      byteIndex >= this.byteLength) {
      return -1;
    }

    // If the buffers have a uniform size, compute the shard directly.
    if (this.bufferUniformSize != null) {
      this.previousShardIndex = Math.floor(byteIndex / this.bufferUniformSize);
      return this.previousShardIndex;
    }

    // If the buffers don't have a uniform size, we need to search for the
    // shard. That means we need a function to check where the byteIndex lies
    // relative to a given shard.
    function check(shard: BufferShard) {
      if (byteIndex < shard.start) {
        return -1;
      }
      if (byteIndex >= shard.end) {
        return 1;
      }
      return 0;
    }

    // For efficiency, try the previous shard first.
    if (check(this.shards[this.previousShardIndex]) === 0) {
      return this.previousShardIndex;
    }

    // Otherwise, use a generic search function.
    // This should almost never end up being used in practice since the weight
    // entries should always be in order.
    const index = search(this.shards, check);
    if (index === -1) {
      return -1;
    }

    this.previousShardIndex = index;
    return this.previousShardIndex;
  }
}

/**
 * Search for an element of a sorted array.
 *
 * @param sortedArray The sorted array to search
 * @param compare A function to compare the current value against the searched
 *     value. Return 0 on a match, negative if the searched value is less than
 *     the value passed to the function, and positive if the searched value is
 *     greater than the value passed to the function.
 * @returns The index of the element, or -1 if it's not in the array.
 */
export function search<T>(sortedArray: T[], compare: (t: T) => number): number {
  // Binary search
  let min = 0;
  let max = sortedArray.length;

  while (min <= max) {
    const middle = Math.floor((max - min) / 2) + min;
    const side = compare(sortedArray[middle]);

    if (side === 0) {
      return middle;
    } else if (side < 0) {
      max = middle;
    } else {
      min = middle + 1;
    }
  }
  return -1;
}
