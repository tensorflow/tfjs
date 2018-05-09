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

// tslint:disable:max-line-length
import {arrayBufferToBase64String, base64StringToArrayBuffer, getModelArtifactsInfoForKerasJSON} from './io_utils';
import {IOHandler, ModelArtifacts, ModelArtifactsInfo, SaveResult} from './types';

// tslint:enable:max-line-length

const PATH_SEPARATOR = '/';
const PATH_PREFIX = 'tensorflowjs_models';
const INFO_SUFFIX = 'info';
const MODEL_TOPOLOGY_SUFFIX = 'model_topology';
const WEIGHT_SPECS_SUFFIX = 'weight_specs';
const WEIGHT_DATA_SUFFIX = 'weight_data';

/**
 * Purge all tensorflow.js-saved model artifacts from local storage.
 *
 * @returns Paths of the models purged.
 */
export function purgeLocalStorageArtifacts(): string[] {
  // TODO(cais): Use central environment flag when it's available.
  if (typeof window === 'undefined' ||
      typeof window.localStorage === 'undefined') {
    throw new Error(
        'purgeLocalStorageModels() cannot proceed because local storage is ' +
        'unavailable in the current environment.');
  }
  const LS = window.localStorage;
  const purgedModelPaths: string[] = [];
  for (let i = 0; i < LS.length; ++i) {
    const key = LS.key(i);
    const prefix = PATH_PREFIX + PATH_SEPARATOR;
    if (key.startsWith(prefix) && key.length > prefix.length) {
      const modelName = key.slice(prefix.length).split(PATH_PREFIX)[0];
      if (purgedModelPaths.indexOf(modelName) === -1) {
        purgedModelPaths.push(modelName);
      }
    }
  }
  return purgedModelPaths;
}

/**
 * IOHandler subclass: Browser Local Storage.
 *
 * See the doc string to `browserLocalStorage` for more details.
 */
export class BrowserLocalStorage implements IOHandler {
  protected readonly LS: Storage;
  protected readonly modelPath: string;
  protected readonly paths: {[key: string]: string};

  constructor(modelPath: string) {
    if (!(window && window.localStorage)) {
      // TODO(cais): Add more info about what IOHandler subtypes are available.
      //   Maybe point to a doc page on the web and/or automatically determine
      //   the available IOHandlers and print them in the error message.
      throw new Error(
          'The current environment does not support local storage.');
    }
    this.LS = window.localStorage;

    if (modelPath == null || !modelPath) {
      throw new Error(
          'For local storage, modelPath must not be null, undefined or empty.');
    }
    this.modelPath = modelPath;
    const modelRoot = [PATH_PREFIX, this.modelPath].join(PATH_SEPARATOR);
    this.paths = {
      info: [modelRoot, INFO_SUFFIX].join(PATH_SEPARATOR),
      topology: [modelRoot, MODEL_TOPOLOGY_SUFFIX].join(PATH_SEPARATOR),
      weightSpecs: [modelRoot, WEIGHT_SPECS_SUFFIX].join(PATH_SEPARATOR),
      weightData: [modelRoot, WEIGHT_DATA_SUFFIX].join(PATH_SEPARATOR),
    };
  }

  /**
   * Save model artifacts to browser local storage.
   *
   * See the documentation to `browserLocalStorage` for details on the saved
   * artifacts.
   *
   * @param modelArtifacts The model artifacts to be stored.
   * @returns An instance of SaveResult.
   */
  async save(modelArtifacts: ModelArtifacts): Promise<SaveResult> {
    if (!(window && window.localStorage)) {
      throw new Error(
          'The current environment does not support local storage.');
    }

    if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
      throw new Error(
          'BrowserLocalStorage.save() does not support saving model topology ' +
          'in binary formats yet.');
    } else {
      const topology = JSON.stringify(modelArtifacts.modelTopology);
      const weightSpecs = JSON.stringify(modelArtifacts.weightSpecs);

      const modelArtifactsInfo: ModelArtifactsInfo =
          getModelArtifactsInfoForKerasJSON(modelArtifacts);

      try {
        this.LS.setItem(this.paths.info, JSON.stringify(modelArtifactsInfo));
        this.LS.setItem(this.paths.topology, topology);
        this.LS.setItem(this.paths.weightSpecs, weightSpecs);
        this.LS.setItem(
            this.paths.weightData,
            arrayBufferToBase64String(modelArtifacts.weightData));

        return {modelArtifactsInfo};
      } catch (err) {
        // If saving failed, clean up all items saved so far.
        for (const key in this.paths) {
          this.LS.removeItem(this.paths[key]);
        }

        throw new Error(
            `Failed to save model '${this.modelPath}' to local storage: ` +
            `size quota being exceeded is a possible cause of this failure: ` +
            `modelTopologyBytes=${modelArtifactsInfo.modelTopologyBytes}, ` +
            `weightSpecsBytes=${modelArtifactsInfo.weightSpecsBytes}, ` +
            `weightDataBytes=${modelArtifactsInfo.weightDataBytes}.`);
      }
    }
  }

  /**
   * Load a model from local storage.
   *
   * See the documentation to `browserLocalStorage` for details on the saved
   * artifacts.
   *
   * @returns The loaded model (if loading succeeds).
   */
  async load(): Promise<ModelArtifacts> {
    const info =
        JSON.parse(this.LS.getItem(this.paths.info)) as ModelArtifactsInfo;
    if (info == null) {
      throw new Error(
          `In local storage, there is no model with name '${this.modelPath}'`);
    }

    if (info.modelTopologyType !== 'KerasJSON') {
      throw new Error(
          'BrowserLocalStorage does not support loading non-JSON model ' +
          'topology yet.');
    }

    const out: ModelArtifacts = {};

    // Load topology.
    const topology = JSON.parse(this.LS.getItem(this.paths.topology));
    if (topology == null) {
      throw new Error(
          `In local storage, the topology of model '${this.modelPath}' ` +
          `is missing.`);
    }
    out.modelTopology = topology;

    // Load weight specs.
    const weightSpecs = JSON.parse(this.LS.getItem(this.paths.weightSpecs));
    if (weightSpecs == null) {
      throw new Error(
          `In local storage, the weight specs of model '${this.modelPath}' ` +
          `are missing.`);
    }
    out.weightSpecs = weightSpecs;

    // Load weight data.
    const weightDataBase64 = this.LS.getItem(this.paths.weightData);
    if (weightDataBase64 == null) {
      throw new Error(
          `In local storage, the binary weight values of model ` +
          `'${this.modelPath}' are missing.`);
    }
    out.weightData = base64StringToArrayBuffer(weightDataBase64);

    return out;
  }
}

/**
 * Factory function for local storage IOHandler.
 *
 * This `IOHandler` supports both `save` and `load`.
 *
 * For each model's saved artifacts, four items are saved to local storage.
 *   - `${PATH_SEPARATOR}/${modelPath}/info`: Contains meta-info about the
 *     model, such as date saved, type of the topology, size in bytes, etc.
 *   - `${PATH_SEPARATOR}/${modelPath}/topology`: Model topology. For Keras-
 *     style models, this is a stringized JSON.
 *   - `${PATH_SEPARATOR}/${modelPath}/weight_specs`: Weight specs of the
 *     model, can be used to decode the saved binary weight values (see
 *     item below).
 *   - `${PATH_SEPARATOR}/${modelPath}/weight_data`: Concatenated binary
 *     weight values, stored as a base64-encoded string.
 *
 * Saving may throw an `Error` if the total size of the artifacts exceed the
 * browser-specific quota.
 *
 * @param modelPath A unique identifier for the model to be saved. Must be a
 *   non-empty string.
 * @returns An instance of `BrowserLocalStorage` (sublcass of `IOHandler`),
 *   which can be used with, e.g., `tf.Model.save`.
 */
export function browserLocalStorage(modelPath: string): IOHandler {
  return new BrowserLocalStorage(modelPath);
}
