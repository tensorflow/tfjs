/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {env} from '../environment';

import {NamedTensorMap} from '../tensor_types';
import {TypedArray} from '../types';
import * as util from '../util';
import {decodeWeights} from './io_utils';
import {monitorPromisesProgress} from './progress';
import {DTYPE_VALUE_SIZE_MAP, LoadOptions, WeightsManifestConfig, WeightsManifestEntry} from './types';

/**
 * Reads binary weights data from a number of URLs.
 *
 * @param fetchURLs URLs to send the HTTP requests at, using `fetch` calls.
 * @param requestOptions RequestInit (options) for the HTTP requests.
 * @param fetchFunc Optional overriding value for the `window.fetch` function.
 * @param onProgress Optional, progress callback function, fired periodically
 *   before the load is completed.
 * @returns A `Promise` of an Array of `ArrayBuffer`. The Array has the same
 *   length as `fetchURLs`.
 */
export async function loadWeightsAsArrayBuffer(
    fetchURLs: string[], loadOptions?: LoadOptions): Promise<ArrayBuffer[]> {
  if (loadOptions == null) {
    loadOptions = {};
  }

  const fetchFunc = loadOptions.fetchFunc == null ? env().platform.fetch :
                                                    loadOptions.fetchFunc;

  // Create the requests for all of the weights in parallel.
  const requests = fetchURLs.map(
      fetchURL =>
          fetchFunc(fetchURL, loadOptions.requestInit, {isBinary: true}));

  const fetchStartFraction = 0;
  const fetchEndFraction = 0.5;

  const responses = loadOptions.onProgress == null ?
      await Promise.all(requests) :
      await monitorPromisesProgress(
          requests, loadOptions.onProgress, fetchStartFraction,
          fetchEndFraction);

  const bufferPromises = responses.map(response => response.arrayBuffer());

  const bufferStartFraction = 0.5;
  const bufferEndFraction = 1;

  const buffers = loadOptions.onProgress == null ?
      await Promise.all(bufferPromises) :
      await monitorPromisesProgress(
          bufferPromises, loadOptions.onProgress, bufferStartFraction,
          bufferEndFraction);
  return buffers;
}

/**
 * Reads a weights manifest JSON configuration, fetches the weights and
 * returns them as `Tensor`s.
 *
 * @param manifest The weights manifest JSON.
 * @param filePathPrefix The path prefix for filenames given in the manifest.
 *     Defaults to the empty string.
 * @param weightNames The names of the weights to be fetched.
 */
export async function loadWeights(
    manifest: WeightsManifestConfig, filePathPrefix = '',
    weightNames?: string[],
    requestInit?: RequestInit): Promise<NamedTensorMap> {
  // TODO(nsthorat): Groups are currently fetched atomically. If you need a
  // single weight from a group, the whole group will be fetched. At a future
  // date, we should support fetching only the individual shards within a
  // group that are needed to reconstruct the requested weight.
  // TODO(cais): Use `decodeWeights` for implementation.

  const fetchWeights = (fetchUrls: string[]) =>
      loadWeightsAsArrayBuffer(fetchUrls, {requestInit});
  const loadWeights = weightsLoaderFactory(fetchWeights);

  return loadWeights(manifest, filePathPrefix, weightNames);
}

/**
 * Creates a function, which reads a weights manifest JSON configuration,
 * fetches the weight files using the specified function and returns them as
 * `Tensor`s.
 *
 * ```js
 * // example for creating a nodejs weight loader, which reads the weight files
 * // from disk using fs.readFileSync
 *
 * import * as fs from 'fs'
 *
 * const fetchWeightsFromDisk = (filePaths: string[]) =>
 *   filePaths.map(filePath => fs.readFileSync(filePath).buffer)
 *
 * const loadWeights = tf.io.weightsLoaderFactory(fetchWeightsFromDisk)
 *
 * const manifest = JSON.parse(
 *   fs.readFileSync('./my_model-weights_manifest').toString()
 * )
 * const weightMap = await loadWeights(manifest, './')
 * ```
 * @param fetchWeightsFunction The function used for fetching the weight files.
 * @returns Weight loading function.
 */
export function weightsLoaderFactory(
    fetchWeightsFunction: (fetchUrls: string[]) => Promise<ArrayBuffer[]>):
    (manifest: WeightsManifestConfig, filePathPrefix?: string,
     weightNames?: string[]) => Promise<NamedTensorMap> {
  return async(
             manifest: WeightsManifestConfig, filePathPrefix = '',
             weightNames?: string[]): Promise<NamedTensorMap> => {
    // Collect all the groups, weights, and their relative offsets to be
    // fetched.
    const groupIndicesToFetchMap = manifest.map(() => false);
    const groupWeightsToFetch: {
      [group: number]: Array<{
        manifestEntry: WeightsManifestEntry; groupOffset: number;
        sizeBytes: number;
      }>
    } = {};
    const weightsFound =
        weightNames != null ? weightNames.map(() => false) : [];
    const allManifestWeightNames: string[] = [];
    manifest.forEach((manifestGroupConfig, groupIndex) => {
      let groupOffset = 0;
      manifestGroupConfig.weights.forEach(weightsEntry => {
        const rawDtype = ('quantization' in weightsEntry) ?
            weightsEntry.quantization.dtype :
            weightsEntry.dtype;

        const weightsBytes = DTYPE_VALUE_SIZE_MAP[rawDtype] *
            util.sizeFromShape(weightsEntry.shape);

        const enqueueWeightsForFetchingFn = () => {
          groupIndicesToFetchMap[groupIndex] = true;
          if (groupWeightsToFetch[groupIndex] == null) {
            groupWeightsToFetch[groupIndex] = [];
          }

          groupWeightsToFetch[groupIndex].push({
            manifestEntry: weightsEntry,
            groupOffset,
            sizeBytes: weightsBytes
          });
        };

        if (weightNames != null) {
          weightNames.forEach((weightName, weightIndex) => {
            if (weightName === weightsEntry.name) {
              enqueueWeightsForFetchingFn();
              weightsFound[weightIndex] = true;
            }
          });
        } else {
          enqueueWeightsForFetchingFn();
        }

        allManifestWeightNames.push(weightsEntry.name);
        groupOffset += weightsBytes;
      });
    });

    if (!weightsFound.every(found => found)) {
      const weightsNotFound = weightNames.filter((_, i) => !weightsFound[i]);
      throw new Error(
          `Could not find weights in manifest with names: ` +
          `${weightsNotFound.join(', ')}. \n` +
          `Manifest JSON has weights with names: ` +
          `${allManifestWeightNames.join(', ')}.`);
    }

    // Convert the one-hot boolean groupId => shouldFetch map to a list of group
    // IDs.
    const groupIndicesToFetch =
        groupIndicesToFetchMap.reduce((accumulator, shouldFetch, i) => {
          if (shouldFetch) {
            accumulator.push(i);
          }
          return accumulator;
        }, []);

    const fetchUrls: string[] = [];
    groupIndicesToFetch.forEach(i => {
      manifest[i].paths.forEach(filepath => {
        const fetchUrl = filePathPrefix +
            (!filePathPrefix.endsWith('/') ? '/' : '') + filepath;
        fetchUrls.push(fetchUrl);
      });
    });
    const buffers = await fetchWeightsFunction(fetchUrls);

    const weightsTensorMap: NamedTensorMap = {};
    let bufferIndexOffset = 0;
    groupIndicesToFetch.forEach(i => {
      const numBuffers = manifest[i].paths.length;

      const weightsBuffer = new CompositeArrayBuffer(
        buffers.slice(bufferIndexOffset, bufferIndexOffset + numBuffers));

      const weightsEntries = groupWeightsToFetch[i];

      weightsEntries.forEach(weightsEntry => {
        const byteBuffer = weightsBuffer.slice(
            weightsEntry.groupOffset,
            weightsEntry.groupOffset + weightsEntry.sizeBytes);
        const nameToTensorMap =
            decodeWeights(byteBuffer, [weightsEntry.manifestEntry]);
        for (const name in nameToTensorMap) {
          weightsTensorMap[name] = nameToTensorMap[name];
        }
      });

      bufferIndexOffset += numBuffers;
    });

    return weightsTensorMap;
  };
}

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

  constructor(buffers: ArrayBuffer | ArrayBuffer[] | TypedArray
      | TypedArray[]) {
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
      this.shards.push({buffer, start, end});
      start = end;
    }

    // Set the byteLenghth
    if (this.shards.length === 0) {
      this.byteLength = 0;
    }
    this.byteLength = this.shards[this.shards.length - 1].end;
  }

  slice(start = 0, end = this.byteLength): ArrayBuffer {
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

      const outputSlice = new Uint8Array(shard.buffer.slice(localStart,
                                                            localEnd));
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
function search<T>(sortedArray: T[], compare: (t: T) => number): number {
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
