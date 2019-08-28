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

import {getModelArtifactsInfoForJSON, toArrayBuffer} from './io_utils';

const stat = promisify(fs.stat);
const writeFile = promisify(fs.writeFile);
const readFile = promisify(fs.readFile);
const mkdir = promisify(fs.mkdir);

function doesNotExistHandler(name: string): (e: NodeJS.ErrnoException) =>
    never {
  return e => {
    switch (e.code) {
      case 'ENOENT':
        throw new Error(`${name} ${e.path} does not exist: loading failed`);
      default:
        throw e;
    }
  };
}

export class NodeFileSystem implements tfc.io.IOHandler {
  static readonly URL_SCHEME = 'file://';

  protected readonly path: string|string[];

  readonly MODEL_JSON_FILENAME = 'model.json';
  readonly WEIGHTS_BINARY_FILENAME = 'weights.bin';
  readonly MODEL_BINARY_FILENAME = 'tensorflowjs.pb';

  /**
   * Constructor of the NodeFileSystem IOHandler.
   * @param path A single path or an Array of paths.
   *   For saving: expects a single path pointing to an existing or nonexistent
   *     directory. If the directory does not exist, it will be
   *     created.
   *   For loading:
   *     - If the model has JSON topology (e.g., `tf.Model`), a single path
   *       pointing to the JSON file (usually named `model.json`) is expected.
   *       The JSON file is expected to contain `modelTopology` and/or
   *       `weightsManifest`. If `weightManifest` exists, the values of the
   *       weights will be loaded from relative paths (relative to the directory
   *       of `model.json`) as contained in `weightManifest`.
   *     - If the model has binary (protocol buffer GraphDef) topology,
   *       an Array of two paths is expected: the first path should point to the
   *       .pb file and the second path should point to the weight manifest
   *       JSON file.
   */
  constructor(path: string|string[]) {
    if (Array.isArray(path)) {
      tfc.util.assert(
          path.length === 2,
          () => 'file paths must have a length of 2, ' +
              `(actual length is ${path.length}).`);
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
      const modelJSON: tfc.io.ModelJSON = {
        modelTopology: modelArtifacts.modelTopology,
        weightsManifest,
        format: modelArtifacts.format,
        generatedBy: modelArtifacts.generatedBy,
        convertedBy: modelArtifacts.convertedBy
      };
      if (modelArtifacts.trainingConfig != null) {
        modelJSON.trainingConfig = modelArtifacts.trainingConfig;
      }
      if (modelArtifacts.userDefinedMetadata != null) {
        modelJSON.userDefinedMetadata = modelArtifacts.userDefinedMetadata;
      }
      const modelJSONPath = join(this.path, this.MODEL_JSON_FILENAME);
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
    return Array.isArray(this.path) ? this.loadBinaryModel() :
                                      this.loadJSONModel();
  }

  protected async loadBinaryModel(): Promise<tfc.io.ModelArtifacts> {
    const topologyPath = this.path[0];
    const weightManifestPath = this.path[1];
    const topology =
        await stat(topologyPath).catch(doesNotExistHandler('Topology Path'));
    const weightManifest =
        await stat(weightManifestPath)
            .catch(doesNotExistHandler('Weight Manifest Path'));

    // `this.path` can be either a directory or a file. If it is a file, assume
    // it is model.json file.
    if (!topology.isFile()) {
      throw new Error('File specified for topology is not a file!');
    }
    if (!weightManifest.isFile()) {
      throw new Error('File specified for the weight manifest is not a file!');
    }

    const modelTopology = await readFile(this.path[0]);
    const weightsManifest = JSON.parse(await readFile(this.path[1], 'utf8'));

    const modelArtifacts: tfc.io.ModelArtifacts = {
      modelTopology,
    };
    const [weightSpecs, weightData] =
        await this.loadWeights(weightsManifest, this.path[1]);

    modelArtifacts.weightSpecs = weightSpecs;
    modelArtifacts.weightData = weightData;

    return modelArtifacts;
  }

  protected async loadJSONModel(): Promise<tfc.io.ModelArtifacts> {
    const path = this.path as string;
    const info = await stat(path).catch(doesNotExistHandler('Path'));

    // `path` can be either a directory or a file. If it is a file, assume
    // it is model.json file.
    if (info.isFile()) {
      const modelJSON = JSON.parse(await readFile(path, 'utf8'));

      const modelArtifacts: tfc.io.ModelArtifacts = {
        modelTopology: modelJSON.modelTopology,
        format: modelJSON.format,
        generatedBy: modelJSON.generatedBy,
        convertedBy: modelJSON.convertedBy
      };
      if (modelJSON.weightsManifest != null) {
        const [weightSpecs, weightData] =
            await this.loadWeights(modelJSON.weightsManifest, path);
        modelArtifacts.weightSpecs = weightSpecs;
        modelArtifacts.weightData = weightData;
      }
      if (modelJSON.trainingConfig != null) {
        modelArtifacts.trainingConfig = modelJSON.trainingConfig;
      }
      if (modelJSON.userDefinedMetadata != null) {
        modelArtifacts.userDefinedMetadata = modelJSON.userDefinedMetadata;
      }
      return modelArtifacts;
    } else {
      throw new Error(
          'The path to load from must be a file. Loading from a directory ' +
          'is not supported.');
    }
  }

  private async loadWeights(
      weightsManifest: tfc.io.WeightsManifestConfig,
      path: string): Promise<[tfc.io.WeightsManifestEntry[], ArrayBuffer]> {
    const dirName = dirname(path);
    const buffers: Buffer[] = [];
    const weightSpecs: tfc.io.WeightsManifestEntry[] = [];
    for (const group of weightsManifest) {
      for (const path of group.paths) {
        const weightFilePath = join(dirName, path);
        const buffer = await readFile(weightFilePath)
                           .catch(doesNotExistHandler('Weight file'));
        buffers.push(buffer);
      }
      weightSpecs.push(...group.weights);
    }
    return [weightSpecs, toArrayBuffer(buffers)];
  }

  /**
   * For each item in `this.path`, creates a directory at the path or verify
   * that the path exists as a directory.
   */
  protected async createOrVerifyDirectory() {
    const paths = Array.isArray(this.path) ? this.path : [this.path];
    for (const path of paths) {
      try {
        await mkdir(path);
      } catch (e) {
        if (e.code === 'EEXIST') {
          if ((await stat(path)).isFile()) {
            throw new Error(
                `Path ${path} exists as a file. The path must be ` +
                `nonexistent or point to a directory.`);
          }
          // else continue, the directory exists
        } else {
          throw e;
        }
      }
    }
  }
}

export const nodeFileSystemRouter = (url: string|string[]) => {
  if (Array.isArray(url)) {
    if (url.every(
            urlElement => urlElement.startsWith(NodeFileSystem.URL_SCHEME))) {
      return new NodeFileSystem(url.map(
          urlElement => urlElement.slice(NodeFileSystem.URL_SCHEME.length)));
    } else {
      return null;
    }
  } else {
    if (url.startsWith(NodeFileSystem.URL_SCHEME)) {
      return new NodeFileSystem(url.slice(NodeFileSystem.URL_SCHEME.length));
    } else {
      return null;
    }
  }
};
// Registration of `nodeFileSystemRouter` is done in index.ts.

/**
 * Factory function for Node.js native file system IO Handler.
 *
 * @param path A single path or an Array of paths.
 *   For saving: expects a single path pointing to an existing or nonexistent
 *     directory. If the directory does not exist, it will be
 *     created.
 *   For loading:
 *     - If the model has JSON topology (e.g., `tf.Model`), a single path
 *       pointing to the JSON file (usually named `model.json`) is expected.
 *       The JSON file is expected to contain `modelTopology` and/or
 *       `weightsManifest`. If `weightManifest` exists, the values of the
 *       weights will be loaded from relative paths (relative to the directory
 *       of `model.json`) as contained in `weightManifest`.
 *     - If the model has binary (protocol buffer GraphDef) topology,
 *       an Array of two paths is expected: the first path should point to the
 *        .pb file and the second path should point to the weight manifest
 *       JSON file.
 */
export function fileSystem(path: string|string[]): NodeFileSystem {
  return new NodeFileSystem(path);
}
