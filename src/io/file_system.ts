/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import * as tfc from '@tensorflow/tfjs-core';
import * as fs from 'fs';
import {dirname, join, resolve} from 'path';
import {promisify} from 'util';

// tslint:disable-next-line:max-line-length
import {getModelArtifactsInfoForJSON, toArrayBuffer} from './io_utils';

export class NodeFileSystem implements tfc.io.IOHandler {
  static readonly URL_SCHEME = 'file://';

  protected readonly path: string|string[];

  readonly MODEL_JSON_FILENAME = 'model.json';
  readonly WEIGHTS_BINARY_FILENAME = 'weights.bin';

  /**
   * Constructor of the NodeFileSystem IOHandler.
   * @param path A single path or an Array of paths.
   *   For saving: expects a single path pointing to an existing or nonexistent
   *     directory. If the directory does not exist, it will be
   *     created. For loading:
   *     - If the model has JSON topology (e.g., `tf.Model`), a single path
   *       pointing to the JSON file (usually named `model.json`) is expected.
   *       The JSON file is expected to contain `modelTopology` and/or
   *       `weightsManifest`. If `weightManifest` exists, the values of the
   *       weights will be loaded from relative paths (relative to the directory
   *       of `model.json`) as contained in `weightManifest`.
   *     - If the model has binary (protocol buffer GraphDef) topology,
   *       an Array of two paths is expected: the first path should point to the
   *       .pb file and the second path should point to the weight manifest
   *       JSON file. This option is *not implemented yet*.
   */
  constructor(path: string|string[]) {
    if (Array.isArray(path)) {
      this.path = path.map(p => resolve(p));
    } else {
      this.path = resolve(path);
    }
  }

  async save(modelArtifacts: tfc.io.ModelArtifacts):
      Promise<tfc.io.SaveResult> {
    if (Array.isArray(this.path)) {
      throw new Error('Cannot perform saving to multiple paths.');
    }

    await this.createOrVerifyDirectory();

    if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
      throw new Error(
          'NodeFileSystem.save() does not support saving model topology ' +
          'in binary format yet.');
      // TODO(cais, nkreeger): Implement this. See
      //   https://github.com/tensorflow/tfjs/issues/343
    } else {
      const weightsBinPath = join(this.path, this.WEIGHTS_BINARY_FILENAME);
      const weightsManifest = [{
        paths: [this.WEIGHTS_BINARY_FILENAME],
        weights: modelArtifacts.weightSpecs
      }];
      const modelJSON = {
        modelTopology: modelArtifacts.modelTopology,
        weightsManifest,
      };
      const modelJSONPath = join(this.path, this.MODEL_JSON_FILENAME);
      const writeFile = promisify(fs.writeFile);
      await writeFile(modelJSONPath, JSON.stringify(modelJSON), 'utf8');
      await writeFile(
          weightsBinPath, Buffer.from(modelArtifacts.weightData), 'binary');

      return {
        // TODO(cais): Use explicit tfc.io.ModelArtifactsInfo type below once it
        // is available.
        // tslint:disable-next-line:no-any
        modelArtifactsInfo: getModelArtifactsInfoForJSON(modelArtifacts) as any
      };
    }
  }

  async load(): Promise<tfc.io.ModelArtifacts> {
    if (Array.isArray(this.path)) {
      throw new Error('Loading from multiple paths is not supported yet.');
      // TODO(cais, nkreeger): Implement this. See
      //   https://github.com/tensorflow/tfjs/issues/343
    }

    const exists = promisify(fs.exists);
    if (!await exists(this.path)) {
      throw new Error(`Path ${this.path} does not exist: loading failed.`);
    }

    // `this.path` can be either a directory or a file. If it is a file, assume
    // it is model.json file.
    const stat = promisify(fs.stat);
    if ((await stat(this.path)).isFile()) {
      const readFile = promisify(fs.readFile);
      const modelJSON = JSON.parse(await readFile(this.path, 'utf8'));

      const modelArtifacts: tfc.io.ModelArtifacts = {
        modelTopology: modelJSON.modelTopology,
      };
      if (modelJSON.weightsManifest != null) {
        const dirName = dirname(this.path);
        const buffers: Buffer[] = [];
        const weightSpecs: tfc.io.WeightsManifestEntry[] = [];
        for (const group of modelJSON.weightsManifest) {
          for (const path of group.paths) {
            const weightFilePath = join(dirName, path);
            if (!await exists(weightFilePath)) {
              throw new Error(`Weight file ${
                  weightFilePath} does not exist: loading failed`);
            }
            const buffer = await readFile(weightFilePath);
            buffers.push(buffer);
          }
          weightSpecs.push(...group.weights);
        }
        modelArtifacts.weightSpecs = weightSpecs;
        modelArtifacts.weightData = toArrayBuffer(buffers);
      }
      return modelArtifacts;
    } else {
      throw new Error(
          'The path to load from must be a file. Loading from a directory ' +
          'is not supported.');
    }
  }

  /**
   * For each item in `this.path`, creates a directory at the path or verify
   * that the path exists as a directory.
   */
  protected async createOrVerifyDirectory() {
    for (const path of Array.isArray(this.path) ? this.path : [this.path]) {
      const exists = promisify(fs.exists);
      const stat = promisify(fs.stat);
      if (await exists(path)) {
        if ((await stat(path)).isFile()) {
          throw new Error(
              `Path ${path} exists as a file. The path must be ` +
              `nonexistent or point to a directory.`);
        }
      } else {
        const mkdir = promisify(fs.mkdir);
        await mkdir(path);
      }
    }
  }
}

export const nodeFileSystemRouter = (url: string) => {
  if (url.startsWith(NodeFileSystem.URL_SCHEME)) {
    return new NodeFileSystem(url.slice(NodeFileSystem.URL_SCHEME.length));
  } else {
    return null;
  }
};
// Registration of `nodeFileSystemRouter` is done in index.ts.
