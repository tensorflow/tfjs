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

import {tensor} from '../ops/ops';
import {NamedTensorMap} from '../tensor_types';
import {TypedArray} from '../types';
import * as util from '../util';
import {DTYPE_VALUE_SIZE_MAP, WeightsManifestConfig, WeightsManifestEntry} from './types';

/**
 * Reads binary weights data from a number of URLs.
 *
 * @param fetchURLs URLs to send the HTTP requests at, using `fetch` calls.
 * @param requestOptions RequestInit (options) for the HTTP requests.
 * @returns A `Promise` of an Array of `ArrayBuffer`. The Array has the same
 *   length as `fetchURLs`.
 */
export async function loadWeightsAsArrayBuffer(
    fetchURLs: string[], requestOptions?: RequestInit): Promise<ArrayBuffer[]> {
  // Create the requests for all of the weights in parallel.
  const requests = fetchURLs.map(fetchURL => fetch(fetchURL, requestOptions));
  const responses = await Promise.all(requests);
  const buffers =
      await Promise.all(responses.map(response => response.arrayBuffer()));
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
    requestOptions?: RequestInit): Promise<NamedTensorMap> {
  // TODO(nsthorat): Groups are currently fetched atomically. If you need a
  // single weight from a group, the whole group will be fetched. At a future
  // date, we should support fetching only the individual shards within a
  // group that are needed to reconstruct the requested weight.
  // TODO(cais): Use `decodeWeights` for implementation.

  // Collect all the groups, weights, and their relative offsets to be
  // fetched.
  const groupIndicesToFetchMap = manifest.map(() => false);
  const groupWeightsToFetch: {
    [group: number]: Array<{
      manifestEntry: WeightsManifestEntry; groupOffset: number;
      sizeBytes: number;
    }>
  } = {};
  const weightsFound = weightNames != null ? weightNames.map(() => false) : [];
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
    const weightsNotFound = weightNames.filter((weight, i) => !weightsFound[i]);
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
  const buffers = await loadWeightsAsArrayBuffer(fetchUrls, requestOptions);

  const weightsTensorMap: NamedTensorMap = {};
  let bufferIndexOffset = 0;
  groupIndicesToFetch.forEach(i => {
    const numBuffers = manifest[i].paths.length;

    let groupBytes = 0;
    for (let i = 0; i < numBuffers; i++) {
      groupBytes += buffers[bufferIndexOffset + i].byteLength;
    }

    // Create a buffer for the whole group.
    const groupBuffer = new ArrayBuffer(groupBytes);
    const groupByteBuffer = new Uint8Array(groupBuffer);
    let groupBufferOffset = 0;
    for (let i = 0; i < numBuffers; i++) {
      const buffer = new Uint8Array(buffers[bufferIndexOffset + i]);
      groupByteBuffer.set(buffer, groupBufferOffset);
      groupBufferOffset += buffer.byteLength;
    }

    const weightsEntries = groupWeightsToFetch[i];

    weightsEntries.forEach(weightsEntry => {
      const byteBuffer = groupBuffer.slice(
          weightsEntry.groupOffset,
          weightsEntry.groupOffset + weightsEntry.sizeBytes);

      let typedArray: TypedArray;

      const dtype = weightsEntry.manifestEntry.dtype;

      if ('quantization' in weightsEntry.manifestEntry) {
        const quantization = weightsEntry.manifestEntry.quantization;
        if (quantization.dtype !== 'uint8' && quantization.dtype !== 'uint16') {
          throw new Error(
              `Weight ${weightsEntry.manifestEntry.name} has unknown ` +
              `quantization dtype ${quantization.dtype}.`);
        }
        const quantizedArray = (quantization.dtype === 'uint8') ?
            new Uint8Array(byteBuffer) :
            new Uint16Array(byteBuffer);
        if (dtype === 'float32') {
          typedArray = Float32Array.from(
              quantizedArray, v => v * quantization.scale + quantization.min);
        } else if (dtype === 'int32') {
          typedArray = Int32Array.from(
              quantizedArray,
              v => Math.round(v * quantization.scale + quantization.min));
        } else {
          throw new Error(
              `Weight ${weightsEntry.manifestEntry.name} has a dtype not ` +
              `supported by quantization: ${dtype}`);
        }
      } else {
        if (dtype === 'float32') {
          typedArray = new Float32Array(byteBuffer);
        } else if (dtype === 'int32') {
          typedArray = new Int32Array(byteBuffer);
        } else if (dtype === 'bool') {
          typedArray = new Uint8Array(byteBuffer);
        } else {
          throw new Error(
              `Weight ${weightsEntry.manifestEntry.name} has unknown dtype ` +
              `${dtype}.`);
        }
      }

      const weightName = weightsEntry.manifestEntry.name;
      if (weightsTensorMap[weightName] != null) {
        throw new Error(
            `Duplicate weight with name ${weightName}. ` +
            `Please make sure weights names are unique in the manifest JSON.`);
      }
      weightsTensorMap[weightName] = tensor(
          typedArray, weightsEntry.manifestEntry.shape,
          weightsEntry.manifestEntry.dtype);
    });

    bufferIndexOffset += numBuffers;
  });

  return weightsTensorMap;
}
