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

/**
 * Classes and functions for model management across multiple storage mediums.
 *
 * Supported client actions:
 * - Listing models on all registered storage mediums.
 * - Remove model by URL from any registered storage mediums, by using URL
 *   string.
 * - Moving or copying model from one path to another in the same medium or from
 *   one medium to another, by using URL strings.
 */

import {assert} from '../util';

import {IORouterRegistry} from './router_registry';
import {ModelArtifactsInfo, ModelStoreManager} from './types';

const URL_SCHEME_SUFFIX = '://';

export class ModelStoreManagerRegistry {
  // Singleton instance.
  private static instance: ModelStoreManagerRegistry;

  private managers: {[scheme: string]: ModelStoreManager};

  private constructor() {
    this.managers = {};
  }

  private static getInstance(): ModelStoreManagerRegistry {
    if (ModelStoreManagerRegistry.instance == null) {
      ModelStoreManagerRegistry.instance = new ModelStoreManagerRegistry();
    }
    return ModelStoreManagerRegistry.instance;
  }

  /**
   * Register a save-handler router.
   *
   * @param saveRouter A function that maps a URL-like string onto an instance
   * of `IOHandler` with the `save` method defined or `null`.
   */
  static registerManager(scheme: string, manager: ModelStoreManager) {
    assert(scheme != null, () => 'scheme must not be undefined or null.');
    if (scheme.endsWith(URL_SCHEME_SUFFIX)) {
      scheme = scheme.slice(0, scheme.indexOf(URL_SCHEME_SUFFIX));
    }
    assert(scheme.length > 0, () => 'scheme must not be an empty string.');
    const registry = ModelStoreManagerRegistry.getInstance();
    assert(
        registry.managers[scheme] == null,
        () => `A model store manager is already registered for scheme '${
            scheme}'.`);
    registry.managers[scheme] = manager;
  }

  static getManager(scheme: string): ModelStoreManager {
    const manager = this.getInstance().managers[scheme];
    if (manager == null) {
      throw new Error(`Cannot find model manager for scheme '${scheme}'`);
    }
    return manager;
  }

  static getSchemes(): string[] {
    return Object.keys(this.getInstance().managers);
  }
}

/**
 * Helper method for parsing a URL string into a scheme and a path.
 *
 * @param url E.g., 'localstorage://my-model'
 * @returns A dictionary with two fields: scheme and path.
 *   Scheme: e.g., 'localstorage' in the example above.
 *   Path: e.g., 'my-model' in the example above.
 */
function parseURL(url: string): {scheme: string, path: string} {
  if (url.indexOf(URL_SCHEME_SUFFIX) === -1) {
    throw new Error(
        `The url string provided does not contain a scheme. ` +
        `Supported schemes are: ` +
        `${ModelStoreManagerRegistry.getSchemes().join(',')}`);
  }
  return {
    scheme: url.split(URL_SCHEME_SUFFIX)[0],
    path: url.split(URL_SCHEME_SUFFIX)[1],
  };
}

async function cloneModelInternal(
    sourceURL: string, destURL: string,
    deleteSource = false): Promise<ModelArtifactsInfo> {
  assert(
      sourceURL !== destURL,
      () => `Old path and new path are the same: '${sourceURL}'`);

  const loadHandlers = IORouterRegistry.getLoadHandlers(sourceURL);
  assert(
      loadHandlers.length > 0,
      () => `Copying failed because no load handler is found for source URL ${
          sourceURL}.`);
  assert(
      loadHandlers.length < 2,
      () => `Copying failed because more than one (${loadHandlers.length}) ` +
          `load handlers for source URL ${sourceURL}.`);
  const loadHandler = loadHandlers[0];

  const saveHandlers = IORouterRegistry.getSaveHandlers(destURL);
  assert(
      saveHandlers.length > 0,
      () => `Copying failed because no save handler is found for destination ` +
          `URL ${destURL}.`);
  assert(
      saveHandlers.length < 2,
      () => `Copying failed because more than one (${loadHandlers.length}) ` +
          `save handlers for destination URL ${destURL}.`);
  const saveHandler = saveHandlers[0];

  const sourceScheme = parseURL(sourceURL).scheme;
  const sourcePath = parseURL(sourceURL).path;
  const sameMedium = sourceScheme === parseURL(sourceURL).scheme;

  const modelArtifacts = await loadHandler.load();

  // If moving within the same storage medium, remove the old model as soon as
  // the loading is done. Without doing this, it is possible that the combined
  // size of the two models will cause the cloning to fail.
  if (deleteSource && sameMedium) {
    await ModelStoreManagerRegistry.getManager(sourceScheme)
        .removeModel(sourcePath);
  }

  const saveResult = await saveHandler.save(modelArtifacts);

  // If moving between mediums, the deletion is done after the save succeeds.
  // This guards against the case in which saving to the destination medium
  // fails.
  if (deleteSource && !sameMedium) {
    await ModelStoreManagerRegistry.getManager(sourceScheme)
        .removeModel(sourcePath);
  }

  return saveResult.modelArtifactsInfo;
}

/**
 * List all models stored in registered storage mediums.
 *
 * For a web browser environment, the registered mediums are Local Storage and
 * IndexedDB.
 *
 * ```js
 * // First create and save a model.
 * const model = tf.sequential();
 * model.add(tf.layers.dense(
 *     {units: 1, inputShape: [10], activation: 'sigmoid'}));
 * await model.save('localstorage://demo/management/model1');
 *
 * // Then list existing models.
 * console.log(JSON.stringify(await tf.io.listModels()));
 *
 * // Delete the model.
 * await tf.io.removeModel('localstorage://demo/management/model1');
 *
 * // List models again.
 * console.log(JSON.stringify(await tf.io.listModels()));
 * ```
 *
 * @returns A `Promise` of a dictionary mapping URLs of existing models to
 * their model artifacts info. URLs include medium-specific schemes, e.g.,
 *   'indexeddb://my/model/1'. Model artifacts info include type of the
 * model's topology, byte sizes of the topology, weights, etc.
 */
/**
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Management',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
async function listModels(): Promise<{[url: string]: ModelArtifactsInfo}> {
  const schemes = ModelStoreManagerRegistry.getSchemes();
  const out: {[url: string]: ModelArtifactsInfo} = {};
  for (const scheme of schemes) {
    const schemeOut =
        await ModelStoreManagerRegistry.getManager(scheme).listModels();
    for (const path in schemeOut) {
      const url = scheme + URL_SCHEME_SUFFIX + path;
      out[url] = schemeOut[path];
    }
  }
  return out;
}

/**
 * Remove a model specified by URL from a reigstered storage medium.
 *
 * ```js
 * // First create and save a model.
 * const model = tf.sequential();
 * model.add(tf.layers.dense(
 *     {units: 1, inputShape: [10], activation: 'sigmoid'}));
 * await model.save('localstorage://demo/management/model1');
 *
 * // Then list existing models.
 * console.log(JSON.stringify(await tf.io.listModels()));
 *
 * // Delete the model.
 * await tf.io.removeModel('localstorage://demo/management/model1');
 *
 * // List models again.
 * console.log(JSON.stringify(await tf.io.listModels()));
 * ```
 *
 * @param url A URL to a stored model, with a scheme prefix, e.g.,
 *   'localstorage://my-model-1', 'indexeddb://my/model/2'.
 * @returns ModelArtifactsInfo of the deleted model (if and only if deletion
 *   is successful).
 * @throws Error if deletion fails, e.g., if no model exists at `path`.
 */
/**
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Management',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
async function removeModel(url: string): Promise<ModelArtifactsInfo> {
  const schemeAndPath = parseURL(url);
  const manager = ModelStoreManagerRegistry.getManager(schemeAndPath.scheme);
  return manager.removeModel(schemeAndPath.path);
}

/**
 * Copy a model from one URL to another.
 *
 * This function supports:
 *
 * 1. Copying within a storage medium, e.g.,
 *    `tf.io.copyModel('localstorage://model-1', 'localstorage://model-2')`
 * 2. Copying between two storage mediums, e.g.,
 *    `tf.io.copyModel('localstorage://model-1', 'indexeddb://model-1')`
 *
 * ```js
 * // First create and save a model.
 * const model = tf.sequential();
 * model.add(tf.layers.dense(
 *     {units: 1, inputShape: [10], activation: 'sigmoid'}));
 * await model.save('localstorage://demo/management/model1');
 *
 * // Then list existing models.
 * console.log(JSON.stringify(await tf.io.listModels()));
 *
 * // Copy the model, from Local Storage to IndexedDB.
 * await tf.io.copyModel(
 *     'localstorage://demo/management/model1',
 *     'indexeddb://demo/management/model1');
 *
 * // List models again.
 * console.log(JSON.stringify(await tf.io.listModels()));
 *
 * // Remove both models.
 * await tf.io.removeModel('localstorage://demo/management/model1');
 * await tf.io.removeModel('indexeddb://demo/management/model1');
 * ```
 *
 * @param sourceURL Source URL of copying.
 * @param destURL Destination URL of copying.
 * @returns ModelArtifactsInfo of the copied model (if and only if copying
 *   is successful).
 * @throws Error if copying fails, e.g., if no model exists at `sourceURL`, or
 *   if `oldPath` and `newPath` are identical.
 */
/**
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Management',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
async function copyModel(
    sourceURL: string, destURL: string): Promise<ModelArtifactsInfo> {
  const deleteSource = false;
  return cloneModelInternal(sourceURL, destURL, deleteSource);
}

/**
 * Move a model from one URL to another.
 *
 * This function supports:
 *
 * 1. Moving within a storage medium, e.g.,
 *    `tf.io.moveModel('localstorage://model-1', 'localstorage://model-2')`
 * 2. Moving between two storage mediums, e.g.,
 *    `tf.io.moveModel('localstorage://model-1', 'indexeddb://model-1')`
 *
 * ```js
 * // First create and save a model.
 * const model = tf.sequential();
 * model.add(tf.layers.dense(
 *     {units: 1, inputShape: [10], activation: 'sigmoid'}));
 * await model.save('localstorage://demo/management/model1');
 *
 * // Then list existing models.
 * console.log(JSON.stringify(await tf.io.listModels()));
 *
 * // Move the model, from Local Storage to IndexedDB.
 * await tf.io.moveModel(
 *     'localstorage://demo/management/model1',
 *     'indexeddb://demo/management/model1');
 *
 * // List models again.
 * console.log(JSON.stringify(await tf.io.listModels()));
 *
 * // Remove the moved model.
 * await tf.io.removeModel('indexeddb://demo/management/model1');
 * ```
 *
 * @param sourceURL Source URL of moving.
 * @param destURL Destination URL of moving.
 * @returns ModelArtifactsInfo of the copied model (if and only if copying
 *   is successful).
 * @throws Error if moving fails, e.g., if no model exists at `sourceURL`, or
 *   if `oldPath` and `newPath` are identical.
 */
/**
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Management',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
async function moveModel(
    sourceURL: string, destURL: string): Promise<ModelArtifactsInfo> {
  const deleteSource = true;
  return cloneModelInternal(sourceURL, destURL, deleteSource);
}

export {moveModel, copyModel, removeModel, listModels};
