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

import {assert} from '../util';
import {concatenateArrayBuffers, getModelArtifactsInfoForJSON} from './io_utils';
import {IORouter, IORouterRegistry} from './router_registry';
import {IOHandler, ModelArtifacts, SaveResult, WeightsManifestConfig, WeightsManifestEntry} from './types';
import {loadWeightsAsArrayBuffer} from './weights_loader';

const OCTET_STREAM_MIME_TYPE = 'application/octet-stream';
const JSON_TYPE = 'application/json';

export class BrowserHTTPRequest implements IOHandler {
  protected readonly path: string|string[];
  protected readonly requestInit: RequestInit;

  private readonly fetchFunc: Function;

  readonly DEFAULT_METHOD = 'POST';

  static readonly URL_SCHEME_REGEX = /^https?:\/\//;

  constructor(
      path: string|string[], requestInit?: RequestInit,
      private readonly weightPathPrefix?: string, fetchFunc?: Function,
      private readonly onProgress?: Function) {
    if (fetchFunc == null) {
      if (typeof fetch === 'undefined') {
        throw new Error(
            'browserHTTPRequest is not supported outside the web browser ' +
            'without a fetch polyfill.');
      }
      // Make sure fetch is always bound to window (the
      // original object) when available.
      fetchFunc = fetch.bind(typeof window === 'undefined' ? null : window);
    } else {
      assert(
          typeof fetchFunc === 'function',
          'Must pass a function that matches the signature of ' +
              '`fetch` (see ' +
              'https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)');
    }

    this.fetchFunc = (path: string, requestInits: RequestInit) => {
      // tslint:disable-next-line:no-any
      return fetchFunc(path, requestInits).catch((error: any) => {
        throw new Error(`Request for ${path} failed due to error: ${error}`);
      });
    };

    assert(
        path != null && path.length > 0,
        'URL path for browserHTTPRequest must not be null, undefined or ' +
            'empty.');

    if (Array.isArray(path)) {
      assert(
          path.length === 2,
          'URL paths for browserHTTPRequest must have a length of 2, ' +
              `(actual length is ${path.length}).`);
    }
    this.path = path;

    if (requestInit != null && requestInit.body != null) {
      throw new Error(
          'requestInit is expected to have no pre-existing body, but has one.');
    }
    this.requestInit = requestInit || {};
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
    const modelTopologyAndWeightManifest = {
      modelTopology: modelArtifacts.modelTopology,
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

    const response = await this.getFetchFunc()(this.path as string, init);

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
    return Array.isArray(this.path) ? this.loadBinaryModel() :
                                      this.loadJSONModel();
  }

  /**
   * Loads the model topology file and build the in memory graph of the model.
   */
  private async loadBinaryTopology(): Promise<ArrayBuffer> {
    const response = await this.getFetchFunc()(this.path[0], this.requestInit);

    if (!response.ok) {
      throw new Error(`Request to ${this.path[0]} failed with error: ${
          response.statusText}`);
    }
    return await response.arrayBuffer();
  }

  protected async loadBinaryModel(): Promise<ModelArtifacts> {
    const graphPromise = this.loadBinaryTopology();
    const manifestPromise =
        await this.getFetchFunc()(this.path[1], this.requestInit);
    if (!manifestPromise.ok) {
      throw new Error(`Request to ${this.path[1]} failed with error: ${
          manifestPromise.statusText}`);
    }

    const results = await Promise.all([graphPromise, manifestPromise]);
    const [modelTopology, weightsManifestResponse] = results;

    const weightsManifest =
        await weightsManifestResponse.json() as WeightsManifestConfig;

    let weightSpecs: WeightsManifestEntry[];
    let weightData: ArrayBuffer;
    if (weightsManifest != null) {
      const results = await this.loadWeights(weightsManifest);
      [weightSpecs, weightData] = results;
    }

    return {modelTopology, weightSpecs, weightData};
  }

  protected async loadJSONModel(): Promise<ModelArtifacts> {
    const modelConfigRequest =
        await this.getFetchFunc()(this.path as string, this.requestInit);

    if (!modelConfigRequest.ok) {
      throw new Error(`Request to ${this.path} failed with error: ${
          modelConfigRequest.statusText}`);
    }
    const modelConfig = await modelConfigRequest.json();
    const modelTopology = modelConfig['modelTopology'];
    const weightsManifest = modelConfig['weightsManifest'];

    // We do not allow both modelTopology and weightsManifest to be missing.
    if (modelTopology == null && weightsManifest == null) {
      throw new Error(
          `The JSON from HTTP path ${this.path} contains neither model ` +
          `topology or manifest for weights.`);
    }

    let weightSpecs: WeightsManifestEntry[];
    let weightData: ArrayBuffer;
    if (weightsManifest != null) {
      const weightsManifest =
          modelConfig['weightsManifest'] as WeightsManifestConfig;
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
    return [
      weightSpecs,
      concatenateArrayBuffers(await loadWeightsAsArrayBuffer(
          fetchURLs, this.requestInit, this.getFetchFunc(), this.onProgress))
    ];
  }

  /**
   * Helper method to get the `fetch`-like function set for this instance.
   *
   * This is mainly for avoiding confusion with regard to what context
   * the `fetch`-like function is bound to. In the default (browser) case,
   * the function will be bound to `window`, instead of `this`.
   */
  private getFetchFunc() {
    return this.fetchFunc;
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
    (url: string|string[], onProgress?: Function) => {
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
          return browserHTTPRequest(url, null, null, null, onProgress);
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
 * The following Python code snippet based on the
 * [flask](https://github.com/pallets/flask) server framework implements a
 * server that can receive the request. Upon receiving the model artifacts
 * via the requst, this particular server reconsistutes instances of
 * [Keras Models](https://keras.io/models/model/) in memory.
 *
 * ```python
 * # pip install -U flask flask-cors tensorflow tensorflowjs
 *
 * from __future__ import absolute_import
 * from __future__ import division
 * from __future__ import print_function
 *
 * import io
 *
 * from flask import Flask, Response, request
 * from flask_cors import CORS, cross_origin
 * import tensorflow as tf
 * import tensorflowjs as tfjs
 * import werkzeug.formparser
 *
 *
 * class ModelReceiver(object):
 *
 *   def __init__(self):
 *     self._model = None
 *     self._model_json_bytes = None
 *     self._model_json_writer = None
 *     self._weight_bytes = None
 *     self._weight_writer = None
 *
 *   @property
 *   def model(self):
 *     self._model_json_writer.flush()
 *     self._weight_writer.flush()
 *     self._model_json_writer.seek(0)
 *     self._weight_writer.seek(0)
 *
 *     json_content = self._model_json_bytes.read()
 *     weights_content = self._weight_bytes.read()
 *     return tfjs.converters.deserialize_keras_model(
 *         json_content,
 *         weight_data=[weights_content],
 *         use_unique_name_scope=True)
 *
 *   def stream_factory(self,
 *                      total_content_length,
 *                      content_type,
 *                      filename,
 *                      content_length=None):
 *     # Note: this example code is *not* thread-safe.
 *     if filename == 'model.json':
 *       self._model_json_bytes = io.BytesIO()
 *       self._model_json_writer = io.BufferedWriter(self._model_json_bytes)
 *       return self._model_json_writer
 *     elif filename == 'model.weights.bin':
 *       self._weight_bytes = io.BytesIO()
 *       self._weight_writer = io.BufferedWriter(self._weight_bytes)
 *       return self._weight_writer
 *
 *
 * def main():
 *   app = Flask('model-server')
 *   CORS(app)
 *   app.config['CORS_HEADER'] = 'Content-Type'
 *
 *   model_receiver = ModelReceiver()
 *
 *   @app.route('/upload', methods=['POST'])
 *   @cross_origin()
 *   def upload():
 *     print('Handling request...')
 *     werkzeug.formparser.parse_form_data(
 *         request.environ, stream_factory=model_receiver.stream_factory)
 *     print('Received model:')
 *     with tf.Graph().as_default(), tf.Session():
 *       model = model_receiver.model
 *       model.summary()
 *       # You can perform `model.predict()`, `model.fit()`,
 *       # `model.evaluate()` etc. here.
 *     return Response(status=200)
 *
 *   app.run('localhost', 5000)
 *
 *
 * if __name__ == '__main__':
 *   main()
 * ```
 *
 * @param path A single URL path or an Array of URL  paths.
 *   Currently, only a single URL path is supported. Array input is reserved
 *   for future development.
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
 * @param weightPathPrefix Optional, this specifies the path prefix for weight
 *   files, by default this is calculated from the path param.
 * @param fetchFunc Optional, custom `fetch` function. E.g., in Node.js,
 *   the `fetch` from node-fetch can be used here.
 * @param onProgress Optional, progress callback function, fired periodically
 *   before the load is completed.
 * @returns An instance of `IOHandler`.
 */
export function browserHTTPRequest(
    path: string|string[], requestInit?: RequestInit, weightPathPrefix?: string,
    fetchFunc?: Function, onProgress?: Function): IOHandler {
  return new BrowserHTTPRequest(
      path, requestInit, weightPathPrefix, fetchFunc, onProgress);
}
