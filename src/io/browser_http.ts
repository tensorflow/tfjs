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
 * IOHandler implementations based on HTTP requests in the web browser.
 *
 * Uses [`fetch`](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API).
 */

import {assert, fetch} from '../util';
import {concatenateArrayBuffers, getModelArtifactsInfoForJSON} from './io_utils';
import {IORouter, IORouterRegistry} from './router_registry';
import {IOHandler, LoadOptions, ModelArtifacts, ModelJSON, OnProgressCallback, SaveResult, WeightsManifestConfig, WeightsManifestEntry} from './types';
import {loadWeightsAsArrayBuffer} from './weights_loader';

const OCTET_STREAM_MIME_TYPE = 'application/octet-stream';
const JSON_TYPE = 'application/json';
export class BrowserHTTPRequest implements IOHandler {
  protected readonly path: string;
  protected readonly requestInit: RequestInit;

  private readonly fetch: Function;

  readonly DEFAULT_METHOD = 'POST';

  static readonly URL_SCHEME_REGEX = /^https?:\/\//;

  private readonly weightPathPrefix: string;
  private readonly onProgress: OnProgressCallback;

  constructor(path: string, loadOptions?: LoadOptions) {
    if (loadOptions == null) {
      loadOptions = {};
    }
    this.weightPathPrefix = loadOptions.weightPathPrefix;
    this.onProgress = loadOptions.onProgress;

    if (loadOptions.fetchFunc != null) {
      assert(
          typeof loadOptions.fetchFunc === 'function',
          () => 'Must pass a function that matches the signature of ' +
              '`fetch` (see ' +
              'https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)');
      this.fetch = loadOptions.fetchFunc;
    } else {
      this.fetch = fetch;
    }

    assert(
        path != null && path.length > 0,
        () =>
            'URL path for browserHTTPRequest must not be null, undefined or ' +
            'empty.');

    if (Array.isArray(path)) {
      assert(
          path.length === 2,
          () => 'URL paths for browserHTTPRequest must have a length of 2, ' +
              `(actual length is ${path.length}).`);
    }
    this.path = path;

    if (loadOptions.requestInit != null &&
        loadOptions.requestInit.body != null) {
      throw new Error(
          'requestInit is expected to have no pre-existing body, but has one.');
    }
    this.requestInit = loadOptions.requestInit || {};
  }

  async save(modelArtifacts: ModelArtifacts): Promise<SaveResult> {
    if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
      throw new Error(
          'BrowserHTTPRequest.save() does not support saving model topology ' +
          'in binary formats yet.');
    }

    const init = Object.assign({method: this.DEFAULT_METHOD}, this.requestInit);
    init.body = new FormData();

    const weightsManifest: WeightsManifestConfig = [{
      paths: ['./model.weights.bin'],
      weights: modelArtifacts.weightSpecs,
    }];
    const modelTopologyAndWeightManifest: ModelJSON = {
      modelTopology: modelArtifacts.modelTopology,
      format: modelArtifacts.format,
      generatedBy: modelArtifacts.generatedBy,
      convertedBy: modelArtifacts.convertedBy,
      weightsManifest
    };

    init.body.append(
        'model.json',
        new Blob(
            [JSON.stringify(modelTopologyAndWeightManifest)],
            {type: JSON_TYPE}),
        'model.json');

    if (modelArtifacts.weightData != null) {
      init.body.append(
          'model.weights.bin',
          new Blob([modelArtifacts.weightData], {type: OCTET_STREAM_MIME_TYPE}),
          'model.weights.bin');
    }

    const response = await this.fetch(this.path, init);

    if (response.ok) {
      return {
        modelArtifactsInfo: getModelArtifactsInfoForJSON(modelArtifacts),
        responses: [response],
      };
    } else {
      throw new Error(
          `BrowserHTTPRequest.save() failed due to HTTP response status ` +
          `${response.status}.`);
    }
  }

  /**
   * Load model artifacts via HTTP request(s).
   *
   * See the documentation to `browserHTTPRequest` for details on the saved
   * artifacts.
   *
   * @returns The loaded model artifacts (if loading succeeds).
   */
  async load(): Promise<ModelArtifacts> {
    const modelConfigRequest = await this.fetch(this.path, this.requestInit);

    if (!modelConfigRequest.ok) {
      throw new Error(
          `Request to ${this.path} failed with status code ` +
          `${modelConfigRequest.status}. Please verify this URL points to ` +
          `the model JSON of the model to load.`);
    }
    let modelConfig: ModelJSON;
    try {
      modelConfig = await modelConfigRequest.json();
    } catch (e) {
      let message = `Failed to parse model JSON of response from ${this.path}.`;
      // TODO(nsthorat): Remove this after some time when we're comfortable that
      // .pb files are mostly gone.
      if (this.path.endsWith('.pb')) {
        message += ' Your path contains a .pb file extension. ' +
            'Support for .pb models have been removed in TensorFlow.js 1.0 ' +
            'in favor of .json models. You can re-convert your Python ' +
            'TensorFlow model using the TensorFlow.js 1.0 conversion scripts ' +
            'or you can convert your.pb models with the \'pb2json\'' +
            'NPM script in the tensorflow/tfjs-converter repository.';
      } else {
        message += ' Please make sure the server is serving valid ' +
            'JSON for this request.';
      }
      throw new Error(message);
    }
    const modelTopology = modelConfig.modelTopology;
    const weightsManifest = modelConfig.weightsManifest;

    // We do not allow both modelTopology and weightsManifest to be missing.
    if (modelTopology == null && weightsManifest == null) {
      throw new Error(
          `The JSON from HTTP path ${this.path} contains neither model ` +
          `topology or manifest for weights.`);
    }

    let weightSpecs: WeightsManifestEntry[];
    let weightData: ArrayBuffer;
    if (weightsManifest != null) {
      const results = await this.loadWeights(weightsManifest);
      [weightSpecs, weightData] = results;
    }

    return {modelTopology, weightSpecs, weightData};
  }

  private async loadWeights(weightsManifest: WeightsManifestConfig):
      Promise<[WeightsManifestEntry[], ArrayBuffer]> {
    const weightPath = Array.isArray(this.path) ? this.path[1] : this.path;
    const [prefix, suffix] = parseUrl(weightPath);
    const pathPrefix = this.weightPathPrefix || prefix;

    const weightSpecs = [];
    for (const entry of weightsManifest) {
      weightSpecs.push(...entry.weights);
    }

    const fetchURLs: string[] = [];
    weightsManifest.forEach(weightsGroup => {
      weightsGroup.paths.forEach(path => {
        fetchURLs.push(pathPrefix + path + suffix);
      });
    });
    const buffers = await loadWeightsAsArrayBuffer(fetchURLs, {
      requestInit: this.requestInit,
      fetchFunc: this.fetch,
      onProgress: this.onProgress
    });
    return [weightSpecs, concatenateArrayBuffers(buffers)];
  }
}

/**
 * Extract the prefix and suffix of the url, where the prefix is the path before
 * the last file, and suffix is the search params after the last file.
 * ```
 * const url = 'http://tfhub.dev/model/1/tensorflowjs_model.pb?tfjs-format=file'
 * [prefix, suffix] = parseUrl(url)
 * // prefix = 'http://tfhub.dev/model/1/'
 * // suffix = '?tfjs-format=file'
 * ```
 * @param url the model url to be parsed.
 */
export function parseUrl(url: string): [string, string] {
  const lastSlash = url.lastIndexOf('/');
  const lastSearchParam = url.lastIndexOf('?');
  const prefix = url.substring(0, lastSlash);
  const suffix =
      lastSearchParam > lastSlash ? url.substring(lastSearchParam) : '';
  return [prefix + '/', suffix];
}

export function isHTTPScheme(url: string): boolean {
  return url.match(BrowserHTTPRequest.URL_SCHEME_REGEX) != null;
}

export const httpRequestRouter: IORouter =
    (url: string, onProgress?: OnProgressCallback) => {
      if (typeof fetch === 'undefined') {
        // browserHTTPRequest uses `fetch`, if one wants to use it in node.js
        // they have to setup a global fetch polyfill.
        return null;
      } else {
        let isHTTP = true;
        if (Array.isArray(url)) {
          isHTTP = url.every(urlItem => isHTTPScheme(urlItem));
        } else {
          isHTTP = isHTTPScheme(url);
        }
        if (isHTTP) {
          return browserHTTPRequest(url, {onProgress});
        }
      }
      return null;
    };
IORouterRegistry.registerSaveRouter(httpRequestRouter);
IORouterRegistry.registerLoadRouter(httpRequestRouter);

/**
 * Creates an IOHandler subtype that sends model artifacts to HTTP server.
 *
 * An HTTP request of the `multipart/form-data` mime type will be sent to the
 * `path` URL. The form data includes artifacts that represent the topology
 * and/or weights of the model. In the case of Keras-style `tf.Model`, two
 * blobs (files) exist in form-data:
 *   - A JSON file consisting of `modelTopology` and `weightsManifest`.
 *   - A binary weights file consisting of the concatenated weight values.
 * These files are in the same format as the one generated by
 * [tfjs_converter](https://js.tensorflow.org/tutorials/import-keras.html).
 *
 * The following code snippet exemplifies the client-side code that uses this
 * function:
 *
 * ```js
 * const model = tf.sequential();
 * model.add(
 *     tf.layers.dense({units: 1, inputShape: [100], activation: 'sigmoid'}));
 *
 * const saveResult = await model.save(tf.io.browserHTTPRequest(
 *     'http://model-server:5000/upload', {method: 'PUT'}));
 * console.log(saveResult);
 * ```
 *
 * If the default `POST` method is to be used, without any custom parameters
 * such as headers, you can simply pass an HTTP or HTTPS URL to `model.save`:
 *
 * ```js
 * const saveResult = await model.save('http://model-server:5000/upload');
 * ```
 *
 * The following GitHub Gist
 * https://gist.github.com/dsmilkov/1b6046fd6132d7408d5257b0976f7864
 * implements a server based on [flask](https://github.com/pallets/flask) that
 * can receive the request. Upon receiving the model artifacts via the requst,
 * this particular server reconsistutes instances of [Keras
 * Models](https://keras.io/models/model/) in memory.
 *
 *
 * @param path A URL path to the model.
 *   Can be an absolute HTTP path (e.g.,
 *   'http://localhost:8000/model-upload)') or a relative path (e.g.,
 *   './model-upload').
 * @param requestInit Request configurations to be used when sending
 *    HTTP request to server using `fetch`. It can contain fields such as
 *    `method`, `credentials`, `headers`, `mode`, etc. See
 *    https://developer.mozilla.org/en-US/docs/Web/API/Request/Request
 *    for more information. `requestInit` must not have a body, because the
 * body will be set by TensorFlow.js. File blobs representing the model
 * topology (filename: 'model.json') and the weights of the model (filename:
 * 'model.weights.bin') will be appended to the body. If `requestInit` has a
 * `body`, an Error will be thrown.
 * @param loadOptions Optional configuration for the loading. It includes the
 *   following fields:
 *   - weightPathPrefix Optional, this specifies the path prefix for weight
 *     files, by default this is calculated from the path param.
 *   - fetchFunc Optional, custom `fetch` function. E.g., in Node.js,
 *     the `fetch` from node-fetch can be used here.
 *   - onProgress Optional, progress callback function, fired periodically
 *     before the load is completed.
 * @returns An instance of `IOHandler`.
 */
/** @doc {heading: 'Models', subheading: 'Loading', namespace: 'io'} */
export function browserHTTPRequest(
    path: string, loadOptions?: LoadOptions): IOHandler {
  return new BrowserHTTPRequest(path, loadOptions);
}
