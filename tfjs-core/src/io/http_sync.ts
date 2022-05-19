/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import {env} from '../environment';
import {Platform} from '../platforms/platform';
import {assert} from '../util';
import {parseUrl} from './http';
import {concatenateArrayBuffers, parseModelJson} from './io_utils';
import {IOHandlerSync, LoadOptions, ModelArtifacts, ModelJSON, SaveResult, WeightsManifestConfig, WeightsManifestEntry} from './types';

type LoadOptionsSync = Omit<LoadOptions, 'weightUrlConverter'|'onProgress'> & {
  weightUrlConverter?: (weightFileName: string) => string;
};

export class HTTPRequestSync implements IOHandlerSync {
  protected readonly path: string;
  protected readonly requestInit: RequestInit;

  private readonly fetch: Function;
  private readonly weightUrlConverter: (weightName: string) => string;

  readonly DEFAULT_METHOD = 'POST';

  static readonly URL_SCHEME_REGEX = /^https?:\/\//;

  private readonly weightPathPrefix: string;

  constructor(path: string, loadOptions?: LoadOptionsSync) {
    if (loadOptions == null) {
      loadOptions = {};
    }
    this.weightPathPrefix = loadOptions.weightPathPrefix;
    this.weightUrlConverter = loadOptions.weightUrlConverter;

    if (loadOptions.fetchFunc != null) {
      assert(
          typeof loadOptions.fetchFunc === 'function',
          () => 'Must pass a function that matches the signature of ' +
              '`fetch` (see ' +
              'https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)');
      this.fetch = loadOptions.fetchFunc;
    } else {
      this.fetch = env().platform.fetchSync;
    }

    assert(
        path != null && path.length > 0,
        () => 'URL path for http must not be null, undefined or ' +
            'empty.');

    if (Array.isArray(path)) {
      assert(
          path.length === 2,
          () => 'URL paths for http must have a length of 2, ' +
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

  save(_modelArtifacts: ModelArtifacts): SaveResult {
    throw new Error('http_sync does not support saving models. Use http instead.');
  }

  /**
   * Load model artifacts via HTTP request(s).
   *
   * See the documentation to `tf.io.http` for details on the saved
   * artifacts.
   *
   * @returns The loaded model artifacts (if loading succeeds).
   */
  load(): ModelArtifacts {
    const modelConfigRequest = this.fetch(this.path, this.requestInit);

    if (!modelConfigRequest.ok) {
      throw new Error(
          `Request to ${this.path} failed with status code ` +
          `${modelConfigRequest.status}. Please verify this URL points to ` +
          `the model JSON of the model to load.`);
    }
    let modelJSON: ModelJSON;
    try {
      modelJSON = modelConfigRequest.json();
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

    // We do not allow both modelTopology and weightsManifest to be missing.
    const modelTopology = modelJSON.modelTopology;
    const weightsManifest = modelJSON.weightsManifest;
    if (modelTopology == null && weightsManifest == null) {
      throw new Error(
          `The JSON from HTTP path ${this.path} contains neither model ` +
          `topology or manifest for weights.`);
    }

    const modelArtifacts = parseModelJson(modelJSON);
    if (modelJSON.weightsManifest != null) {
      const [weightSpecs, weightData] =
        this.loadWeights(modelJSON.weightsManifest);
      modelArtifacts.weightSpecs = weightSpecs;
      modelArtifacts.weightData = weightData;

    }
    return modelArtifacts;
  }

  private loadWeights(weightsManifest: WeightsManifestConfig):
      [WeightsManifestEntry[], ArrayBuffer] {
    const weightPath = Array.isArray(this.path) ? this.path[1] : this.path;
    const [prefix, suffix] = parseUrl(weightPath);
    const pathPrefix = this.weightPathPrefix || prefix;

    const weightSpecs = [];
    for (const entry of weightsManifest) {
      weightSpecs.push(...entry.weights);
    }

    const fetchURLs: string[] = [];
    const urls: string[] = [];
    for (const weightsGroup of weightsManifest) {
      for (const path of weightsGroup.paths) {
        if (this.weightUrlConverter != null) {
          urls.push(this.weightUrlConverter(path));
        } else {
          fetchURLs.push(pathPrefix + path + suffix);
        }
      }
    }

    if (this.weightUrlConverter) {
      fetchURLs.push(...urls);
    }

    const buffers = loadWeightsAsArrayBufferSync(
      fetchURLs, this.fetch as Platform['fetchSync'], this.requestInit);
    return [weightSpecs, concatenateArrayBuffers(buffers)];
  }
}

export function httpSync(path: string, loadOptions?: LoadOptionsSync):
IOHandlerSync {
  return new HTTPRequestSync(path, loadOptions);
}

/**
 * Synchronously reads binary weights data from a number of URLs.
 *
 * @param fetchURLs URLs to send the HTTP requests at, using synchronous
 * `fetch` calls.
 * @param requestOptions RequestInit (options) for the HTTP requests.
 * @returns An Array of `ArrayBuffer`. The Array has the same
 *   length as `fetchURLs`.
 */
function loadWeightsAsArrayBufferSync(
  fetchURLs: string[], fetchFunction: Platform['fetchSync'],
  requestOptions: RequestInit = {}): ArrayBuffer[] {
  const responses = fetchURLs.map(
      fetchURL =>
      fetchFunction(fetchURL, requestOptions, {isBinary: true}));
  return responses.map(response => response.arrayBuffer());
}
