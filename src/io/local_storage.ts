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

import {ENV} from '../environment';
import {assert} from '../util';
import {arrayBufferToBase64String, base64StringToArrayBuffer, getModelArtifactsInfoForJSON} from './io_utils';
import {ModelStoreManagerRegistry} from './model_management';
import {IORouter, IORouterRegistry} from './router_registry';
import {IOHandler, ModelArtifacts, ModelArtifactsInfo, ModelStoreManager, SaveResult} from './types';

const PATH_SEPARATOR = '/';
const PATH_PREFIX = 'tensorflowjs_models';
const INFO_SUFFIX = 'info';
const MODEL_TOPOLOGY_SUFFIX = 'model_topology';
const WEIGHT_SPECS_SUFFIX = 'weight_specs';
const WEIGHT_DATA_SUFFIX = 'weight_data';
const MODEL_METADATA_SUFFIX = 'model_metadata';

/**
 * Purge all tensorflow.js-saved model artifacts from local storage.
 *
 * @returns Paths of the models purged.
 */
export function purgeLocalStorageArtifacts(): string[] {
  if (!ENV.get('IS_BROWSER') || typeof window.localStorage === 'undefined') {
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
      LS.removeItem(key);
      const modelName = getModelPathFromKey(key);
      if (purgedModelPaths.indexOf(modelName) === -1) {
        purgedModelPaths.push(modelName);
      }
    }
  }
  return purgedModelPaths;
}

function getModelKeys(path: string): {
  info: string,
  topology: string,
  weightSpecs: string,
  weightData: string,
  modelMetadata: string
} {
  return {
    info: [PATH_PREFIX, path, INFO_SUFFIX].join(PATH_SEPARATOR),
    topology: [PATH_PREFIX, path, MODEL_TOPOLOGY_SUFFIX].join(PATH_SEPARATOR),
    weightSpecs: [PATH_PREFIX, path, WEIGHT_SPECS_SUFFIX].join(PATH_SEPARATOR),
    weightData: [PATH_PREFIX, path, WEIGHT_DATA_SUFFIX].join(PATH_SEPARATOR),
    modelMetadata:
        [PATH_PREFIX, path, MODEL_METADATA_SUFFIX].join(PATH_SEPARATOR)
  };
}

/**
 * Get model path from a local-storage key.
 *
 * E.g., 'tensorflowjs_models/my/model/1/info' --> 'my/model/1'
 *
 * @param key
 */
function getModelPathFromKey(key: string) {
  const items = key.split(PATH_SEPARATOR);
  if (items.length < 3) {
    throw new Error(`Invalid key format: ${key}`);
  }
  return items.slice(1, items.length - 1).join(PATH_SEPARATOR);
}

function maybeStripScheme(key: string) {
  return key.startsWith(BrowserLocalStorage.URL_SCHEME) ?
      key.slice(BrowserLocalStorage.URL_SCHEME.length) :
      key;
}

declare type LocalStorageKeys = {
  info: string,
  topology: string,
  weightSpecs: string,
  weightData: string,
  modelMetadata: string
};

/**
 * IOHandler subclass: Browser Local Storage.
 *
 * See the doc string to `browserLocalStorage` for more details.
 */
export class BrowserLocalStorage implements IOHandler {
  protected readonly LS: Storage;
  protected readonly modelPath: string;
  protected readonly keys: LocalStorageKeys;

  static readonly URL_SCHEME = 'localstorage://';

  constructor(modelPath: string) {
    if (!ENV.get('IS_BROWSER') || typeof window.localStorage === 'undefined') {
      // TODO(cais): Add more info about what IOHandler subtypes are
      // available.
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
    this.keys = getModelKeys(this.modelPath);
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
    if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
      throw new Error(
          'BrowserLocalStorage.save() does not support saving model topology ' +
          'in binary formats yet.');
    } else {
      const topology = JSON.stringify(modelArtifacts.modelTopology);
      const weightSpecs = JSON.stringify(modelArtifacts.weightSpecs);

      const modelArtifactsInfo: ModelArtifactsInfo =
          getModelArtifactsInfoForJSON(modelArtifacts);

      try {
        this.LS.setItem(this.keys.info, JSON.stringify(modelArtifactsInfo));
        this.LS.setItem(this.keys.topology, topology);
        this.LS.setItem(this.keys.weightSpecs, weightSpecs);
        this.LS.setItem(
            this.keys.weightData,
            arrayBufferToBase64String(modelArtifacts.weightData));
        this.LS.setItem(this.keys.modelMetadata, JSON.stringify({
          format: modelArtifacts.format,
          generatedBy: modelArtifacts.generatedBy,
          convertedBy: modelArtifacts.convertedBy
        }));

        return {modelArtifactsInfo};
      } catch (err) {
        // If saving failed, clean up all items saved so far.
        this.LS.removeItem(this.keys.info);
        this.LS.removeItem(this.keys.topology);
        this.LS.removeItem(this.keys.weightSpecs);
        this.LS.removeItem(this.keys.weightData);
        this.LS.removeItem(this.keys.modelMetadata);

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
        JSON.parse(this.LS.getItem(this.keys.info)) as ModelArtifactsInfo;
    if (info == null) {
      throw new Error(
          `In local storage, there is no model with name '${this.modelPath}'`);
    }

    if (info.modelTopologyType !== 'JSON') {
      throw new Error(
          'BrowserLocalStorage does not support loading non-JSON model ' +
          'topology yet.');
    }

    const out: ModelArtifacts = {};

    // Load topology.
    const topology = JSON.parse(this.LS.getItem(this.keys.topology));
    if (topology == null) {
      throw new Error(
          `In local storage, the topology of model '${this.modelPath}' ` +
          `is missing.`);
    }
    out.modelTopology = topology;

    // Load weight specs.
    const weightSpecs = JSON.parse(this.LS.getItem(this.keys.weightSpecs));
    if (weightSpecs == null) {
      throw new Error(
          `In local storage, the weight specs of model '${this.modelPath}' ` +
          `are missing.`);
    }
    out.weightSpecs = weightSpecs;

    // Load meta-data fields.
    const metadataString = this.LS.getItem(this.keys.modelMetadata);
    if (metadataString != null) {
      const metadata = JSON.parse(metadataString) as
          {format: string, generatedBy: string, convertedBy: string};
      out.format = metadata['format'];
      out.generatedBy = metadata['generatedBy'];
      out.convertedBy = metadata['convertedBy'];
    }

    // Load weight data.
    const weightDataBase64 = this.LS.getItem(this.keys.weightData);
    if (weightDataBase64 == null) {
      throw new Error(
          `In local storage, the binary weight values of model ` +
          `'${this.modelPath}' are missing.`);
    }
    out.weightData = base64StringToArrayBuffer(weightDataBase64);

    return out;
  }
}

export const localStorageRouter: IORouter = (url: string|string[]) => {
  if (!ENV.get('IS_BROWSER')) {
    return null;
  } else {
    if (!Array.isArray(url) && url.startsWith(BrowserLocalStorage.URL_SCHEME)) {
      return browserLocalStorage(
          url.slice(BrowserLocalStorage.URL_SCHEME.length));
    } else {
      return null;
    }
  }
};
IORouterRegistry.registerSaveRouter(localStorageRouter);
IORouterRegistry.registerLoadRouter(localStorageRouter);

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
 * @returns An instance of `IOHandler`, which can be used with, e.g.,
 *   `tf.Model.save`.
 */
export function browserLocalStorage(modelPath: string): IOHandler {
  return new BrowserLocalStorage(modelPath);
}

export class BrowserLocalStorageManager implements ModelStoreManager {
  private readonly LS: Storage;

  constructor() {
    assert(
        ENV.get('IS_BROWSER'),
        () => 'Current environment is not a web browser');
    assert(
        typeof window.localStorage !== 'undefined',
        () => 'Current browser does not appear to support localStorage');
    this.LS = window.localStorage;
  }

  async listModels(): Promise<{[path: string]: ModelArtifactsInfo}> {
    const out: {[path: string]: ModelArtifactsInfo} = {};
    const prefix = PATH_PREFIX + PATH_SEPARATOR;
    const suffix = PATH_SEPARATOR + INFO_SUFFIX;
    for (let i = 0; i < this.LS.length; ++i) {
      const key = this.LS.key(i);
      if (key.startsWith(prefix) && key.endsWith(suffix)) {
        const modelPath = getModelPathFromKey(key);
        out[modelPath] = JSON.parse(this.LS.getItem(key)) as ModelArtifactsInfo;
      }
    }
    return out;
  }

  async removeModel(path: string): Promise<ModelArtifactsInfo> {
    path = maybeStripScheme(path);
    const keys = getModelKeys(path);
    if (this.LS.getItem(keys.info) == null) {
      throw new Error(`Cannot find model at path '${path}'`);
    }
    const info = JSON.parse(this.LS.getItem(keys.info)) as ModelArtifactsInfo;

    this.LS.removeItem(keys.info);
    this.LS.removeItem(keys.topology);
    this.LS.removeItem(keys.weightSpecs);
    this.LS.removeItem(keys.weightData);
    return info;
  }
}

if (ENV.get('IS_BROWSER')) {
  // Wrap the construction and registration, to guard against browsers that
  // don't support Local Storage.
  try {
    ModelStoreManagerRegistry.registerManager(
        BrowserLocalStorage.URL_SCHEME, new BrowserLocalStorageManager());
  } catch (err) {
  }
}
