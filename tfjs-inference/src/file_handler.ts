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

import * as tf from '@tensorflow/tfjs-core';
import * as fs from 'fs';
import {dirname, join, resolve} from 'path';

/**
 * Handler for loading model files from local file system.
 */
export class FileHandler implements tf.io.IOHandler {
  protected readonly path: string;

  /**
   * Constructor of the FileHandler.
   * @param path A single path pointing to the JSON file (usually named
   *     `model.json`) is expected. The JSON file is expected to contain
   *     `modelTopology` and/or `weightsManifest`. If `weightsManifest` exists,
   *     the values of the weights will be loaded from relative paths (relative
   *     to the directory of `model.json`) as contained in `weightManifest`.
   */
  constructor(path: string) {
    this.path = resolve(path);
  }

  /**
   * Load a tfjs model.
   *
   *  @return A promise of tfjs model.
   */
  async load(): Promise<tf.io.ModelArtifacts> {
    const path = this.path;

    const fsInfo = fs.statSync(path);

    if (!fsInfo.isFile()) {
      throw new Error(
          'The path to load from must be a file. Loading from a directory ' +
          'is not supported.');
    } else {
      const file = fs.readFileSync(path, 'utf8');
      const modelJSON = JSON.parse(file);

      // Mapping modelJSON to modelArtifacts.
      const modelArtifacts: tf.io.ModelArtifacts = {
        modelTopology: modelJSON.modelTopology,
        format: modelJSON.format,
        generatedBy: modelJSON.generatedBy,
        convertedBy: modelJSON.convertedBy
      };

      if (modelJSON.signature != null) {
        modelArtifacts.signature = modelJSON.signature;
      }

      if (modelJSON.userDefinedMetadata != null) {
        modelArtifacts.userDefinedMetadata = modelJSON.userDefinedMetadata;
      }

      if (modelJSON.modelInitializer != null) {
        modelArtifacts.modelInitializer = modelJSON.modelInitializer;
      }

      if (modelJSON.weightsManifest != null) {
        const [weightSpecs, weightData] =
            this.loadWeights(modelJSON.weightsManifest, path);
        modelArtifacts.weightSpecs = weightSpecs;
        modelArtifacts.weightData = weightData;
      }
      if (modelJSON.trainingConfig != null) {
        modelArtifacts.trainingConfig = modelJSON.trainingConfig;
      }

      return modelArtifacts;
    }
  }

  /**
   * Load weights binary files from a local path.
   *
   * @param weightsManifest Config file that has
   *     information of locations of the weight binary files.
   * @param path Path of the `model.json` file, the weights binary
   *     files are expected to be in the same directory.
   * @return [weightSpecs, weightData].
   */
  private loadWeights(
      weightsManifest: tf.io.WeightsManifestConfig,
      path: string): [tf.io.WeightsManifestEntry[], ArrayBuffer] {
    const dirName = dirname(path);
    const buffers: Buffer[] = [];
    const weightSpecs: tf.io.WeightsManifestEntry[] = [];
    for (const group of weightsManifest) {
      for (const path of group.paths) {
        const weightFilePath = join(dirName, path);
        const buffer = fs.readFileSync(weightFilePath);
        buffers.push(buffer);
      }
      weightSpecs.push(...group.weights);
    }
    return [weightSpecs, toArrayBuffer(buffers)];
  }
}

/**
 * Convert a Buffer or an Array of Buffers to an ArrayBuffer.
 *
 * If the input is an Array of Buffers, they will be concatenated in the
 * specified order to form the output ArrayBuffer.
 *
 * @param buf A Buffer or an Array of Buffers.
 * @return An ArrayBuffer.
 */
function toArrayBuffer(buf: Buffer|Buffer[]): ArrayBuffer {
  if (Array.isArray(buf)) {
    const newBuf = Buffer.concat(buf);

    return newBuf.buffer.slice(
        newBuf.byteOffset, newBuf.byteOffset + newBuf.byteLength);
  } else {
    // A single Buffer. Return a copy of the underlying ArrayBuffer slice.
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  }
}
