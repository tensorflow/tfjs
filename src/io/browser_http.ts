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

export class BrowserHTTPRequest implements IOHandler {
  protected readonly path: string;
  protected readonly requestInit: RequestInit;

  readonly DEFAULT_METHOD = 'POST';

  static readonly URL_SCHEMES = ['http://', 'https://'];

  constructor(path: string, requestInit?: RequestInit) {
    if (typeof fetch === 'undefined') {
      throw new Error(
          // tslint:disable-next-line:max-line-length
          'browserHTTPRequest is not supported outside the web browser without a fetch polyfill.');
    }

    assert(
        path != null && path.length > 0,
        'URL path for browserHTTPRequest must not be null, undefined or ' +
            'empty.');
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
            {type: 'application/json'}),
        'model.json');

    if (modelArtifacts.weightData != null) {
      init.body.append(
          'model.weights.bin',
          new Blob(
              [modelArtifacts.weightData], {type: 'application/octet-stream'}),
          'model.weights.bin');
    }

    const response = await fetch(this.path, init);

    if (response.status === 200) {
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
    const modelConfigRequest = await fetch(this.path, this.requestInit);
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
      weightSpecs = [];
      for (const entry of weightsManifest) {
        weightSpecs.push(...entry.weights);
      }

      let pathPrefix = this.path.substring(0, this.path.lastIndexOf('/'));
      if (!pathPrefix.endsWith('/')) {
        pathPrefix = pathPrefix + '/';
      }

      const fetchURLs: string[] = [];
      weightsManifest.forEach(weightsGroup => {
        weightsGroup.paths.forEach(path => {
          fetchURLs.push(pathPrefix + path);
        });
      });
      weightData = concatenateArrayBuffers(
          await loadWeightsAsArrayBuffer(fetchURLs, this.requestInit));
    }

    return {modelTopology, weightSpecs, weightData};
  }
}

export const httpRequestRouter: IORouter = (url: string) => {
  if (typeof fetch === 'undefined') {
    // browserHTTPRequest uses `fetch`, if one wants to use it in node.js
    // they have to setup a global fetch polyfill.
    return null;
  } else {
    for (const scheme of BrowserHTTPRequest.URL_SCHEMES) {
      if (url.startsWith(scheme)) {
        return browserHTTPRequest(url);
      }
    }
    return null;
  }
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
 * @param path URL path. Can be an absolute HTTP path (e.g.,
 *   'http://localhost:8000/model-upload)') or a relative path (e.g.,
 *   './model-upload').
 * @param requestInit Request configurations to be used when sending
 *    HTTP request to server using `fetch`. It can contain fields such as
 *    `method`, `credentials`, `headers`, `mode`, etc. See
 *    https://developer.mozilla.org/en-US/docs/Web/API/Request/Request
 *    for more information. `requestInit` must not have a body, because the body
 *    will be set by TensorFlow.js. File blobs representing
 *    the model topology (filename: 'model.json') and the weights of the
 *    model (filename: 'model.weights.bin') will be appended to the body.
 *    If `requestInit` has a `body`, an Error will be thrown.
 * @returns An instance of `IOHandler`.
 */
export function browserHTTPRequest(
    path: string, requestInit?: RequestInit): IOHandler {
  return new BrowserHTTPRequest(path, requestInit);
}
