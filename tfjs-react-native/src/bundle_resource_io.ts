/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {io, util} from '@tensorflow/tfjs-core';
import {Asset} from 'expo-asset';
import {Platform} from 'react-native';

import {fetch} from './platform_react_native';

class BundleResourceHandler implements io.IOHandler {
  constructor(
      protected readonly modelJson: io.ModelJSON,
      protected readonly modelWeightsId: string|number) {
    if (modelJson == null || modelWeightsId == null) {
      throw new Error(
          'Must pass the model json object and the model weights path.');
    }
    if (Array.isArray(modelWeightsId)) {
      throw new Error(
          'Bundle resource IO handler does not currently support loading ' +
          'sharded weights');
    }
  }

  /**
   * Save model artifacts. This IO handler cannot support writing to the
   * packaged bundle at runtime and is exclusively for loading a model
   * that is already packages with the app.
   */
  async save(): Promise<io.SaveResult> {
    throw new Error(
        'Bundle resource IO handler does not support saving. ' +
        'Consider using asyncStorageIO instead');
  }

  /**
   * Load a model from local storage.
   *
   * See the documentation to `browserLocalStorage` for details on the saved
   * artifacts.
   *
   * @returns The loaded model (if loading succeeds).
   */
  async load(): Promise<io.ModelArtifacts> {
    const modelJson = this.modelJson;

    if (modelJson.weightsManifest.length > 1) {
      throw new Error(
          'Bundle resource IO handler does not currently support loading ' +
          'sharded weights and the modelJson indicates that this model has ' +
          'sharded weights (more than one weights file).');
    }

    const weightsAsset = Asset.fromModule(this.modelWeightsId);
    if (weightsAsset.uri.match('^http')) {
      // In debug/dev mode RN will serve these assets over HTTP
      return this.loadViaHttp(weightsAsset);
    } else {
      // In release mode the assets will be on the file system.
      return this.loadLocalAsset(weightsAsset);
    }
  }

  async loadViaHttp(weightsAsset: Asset): Promise<io.ModelArtifacts> {
    const modelJson = this.modelJson;

    // Load the weights
    const url = weightsAsset.uri;
    const requestInit: undefined = undefined;
    const response = await fetch(url, requestInit, {isBinary: true});
    const weightData = await response.arrayBuffer();

    const modelArtifacts: io.ModelArtifacts = Object.assign({}, modelJson);
    modelArtifacts.weightSpecs = modelJson.weightsManifest[0].weights;
    //@ts-ignore
    delete modelArtifacts.weightManifest;
    modelArtifacts.weightData = weightData;
    return modelArtifacts;
  }

  async loadLocalAsset(weightsAsset: Asset): Promise<io.ModelArtifacts> {
    // Use a dynamic import here because react-native-fs is not compatible
    // with managed expo workflow. However the managed expo workflow should
    // never hit this code path.

    // tslint:disable-next-line: no-require-imports
    const RNFS = require('react-native-fs');

    const modelJson = this.modelJson;

    let base64Weights: string;
    if (Platform.OS === 'android') {
      // On android we get a resource id instead of a regular path. We need
      // to load the weights from the res/raw folder using this id.
      const fileName = `${weightsAsset.uri}.${weightsAsset.type}`;
      try {
        base64Weights = await RNFS.readFileRes(fileName, 'base64');
      } catch (e) {
        throw new Error(
            `Error reading resource ${fileName}. Make sure the file is
            in located in the res/raw folder of the bundle`,
        );
      }
    } else {
      try {
        base64Weights = await RNFS.readFile(weightsAsset.uri, 'base64');
      } catch (e) {
        throw new Error(
            `Error reading resource ${weightsAsset.uri}.`,
        );
      }
    }

    const weightData = util.encodeString(base64Weights, 'base64').buffer;
    const modelArtifacts: io.ModelArtifacts = Object.assign({}, modelJson);
    modelArtifacts.weightSpecs = modelJson.weightsManifest[0].weights;
    //@ts-ignore
    delete modelArtifacts.weightManifest;
    modelArtifacts.weightData = weightData;
    return modelArtifacts;
  }
}

/**
 * Factory function for BundleResource IOHandler.
 *
 * This `IOHandler` only supports `load`. It is designed to support
 * loading models that have been statically bundled (at compile time)
 * with an app.
 *
 * This IOHandler is not compatible with managed expo apps.
 *
 * ```js
 *  const modelJson = require('../path/to/model.json');
 *  const modelWeights = require('../path/to/model_weights.bin');
 *  async function bundleResourceIOExample() {
 *    const model =
 *      await tf.loadLayersModel(bundleResourceIO(modelJson, modelWeights));
 *
 *     const res = model.predict(tf.randomNormal([1, 28, 28, 1])) as tf.Tensor;
 *  }
 * ```
 *
 * @param modelJson The JSON object for the serialized model.
 * @param modelWeightsId An identifier for the model weights file. This is
 * generally a resourceId or a path to the resource in the app package.
 * This is typically obtained with a `require` statement.
 *
 * See
 * facebook.github.io/react-native/docs/images#static-non-image-resources
 * for more details on how to include static resources into your react-native
 * app including how to configure `metro` to bundle `.bin` files.
 *
 * @returns An instance of `IOHandler`
 *
 * @doc {heading: 'Models', subheading: 'IOHandlers'}
 */
export function bundleResourceIO(
    modelJson: io.ModelJSON, modelWeightsId: number): io.IOHandler {
  if (typeof modelJson !== 'object') {
    throw new Error(
        'modelJson must be a JavaScript object (and not a string).\n' +
        'Have you wrapped yor asset path in a require() statment?');
  }

  if (typeof modelWeightsId !== 'number') {
    throw new Error(
        'modelWeightsID must be a number.\n' +
        'Have you wrapped yor asset path in a require() statment?');
  }
  return new BundleResourceHandler(modelJson, modelWeightsId);
}
