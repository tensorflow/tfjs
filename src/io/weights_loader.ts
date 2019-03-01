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

import {NamedTensorMap} from '../tensor_types';
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

  const fetchFunc =
      loadOptions.fetchFunc == null ? fetch : loadOptions.fetchFunc;

  // Create the requests for all of the weights in parallel.
  const requests =
      fetchURLs.map(fetchURL => fetchFunc(fetchURL, loadOptions.requestInit));

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
