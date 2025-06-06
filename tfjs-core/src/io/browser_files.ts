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
 * IOHandlers related to files, such as browser-triggered file downloads,
 * user-selected files in browser.
 */

import '../flags';
import {env} from '../environment';

import {basename, getModelArtifactsForJSON, getModelArtifactsInfoForJSON, getModelJSONForModelArtifacts} from './io_utils';
import {IORouter, IORouterRegistry} from './router_registry';
import {IOHandler, ModelArtifacts, ModelJSON, SaveResult, WeightData, WeightsManifestConfig, WeightsManifestEntry} from './types';
import {CompositeArrayBuffer} from './composite_array_buffer';

const DEFAULT_FILE_NAME_PREFIX = 'model';
const DEFAULT_JSON_EXTENSION_NAME = '.json';
const DEFAULT_WEIGHT_DATA_EXTENSION_NAME = '.weights.bin';

function defer<T>(f: () => T): Promise<T> {
  return new Promise(resolve => setTimeout(resolve)).then(f);
}

export class BrowserDownloads implements IOHandler {
  private readonly modelJsonFileName: string;
  private readonly weightDataFileName: string;
  private readonly modelJsonAnchor: HTMLAnchorElement;
  private readonly weightDataAnchor: HTMLAnchorElement;

  static readonly URL_SCHEME = 'downloads://';

  constructor(fileNamePrefix?: string) {
    if (!env().getBool('IS_BROWSER')) {
      // TODO(cais): Provide info on what IOHandlers are available under the
      //   current environment.
      throw new Error(
          'browserDownloads() cannot proceed because the current environment ' +
          'is not a browser.');
    }

    if (fileNamePrefix.startsWith(BrowserDownloads.URL_SCHEME)) {
      fileNamePrefix = fileNamePrefix.slice(BrowserDownloads.URL_SCHEME.length);
    }
    if (fileNamePrefix == null || fileNamePrefix.length === 0) {
      fileNamePrefix = DEFAULT_FILE_NAME_PREFIX;
    }

    this.modelJsonFileName = fileNamePrefix + DEFAULT_JSON_EXTENSION_NAME;
    this.weightDataFileName =
        fileNamePrefix + DEFAULT_WEIGHT_DATA_EXTENSION_NAME;
  }

  async save(modelArtifacts: ModelArtifacts): Promise<SaveResult> {
    if (typeof (document) === 'undefined') {
      throw new Error(
          'Browser downloads are not supported in ' +
          'this environment since `document` is not present');
    }

    // TODO(mattsoulanille): Support saving models over 2GB that exceed
    // Chrome's ArrayBuffer size limit.
    const weightBuffer = CompositeArrayBuffer.join(modelArtifacts.weightData);

    const weightsURL = window.URL.createObjectURL(new Blob(
        [weightBuffer], {type: 'application/octet-stream'}));

    if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
      throw new Error(
          'BrowserDownloads.save() does not support saving model topology ' +
          'in binary formats yet.');
    } else {
      const weightsManifest: WeightsManifestConfig = [{
        paths: ['./' + this.weightDataFileName],
        weights: modelArtifacts.weightSpecs
      }];
      const modelJSON: ModelJSON =
          getModelJSONForModelArtifacts(modelArtifacts, weightsManifest);

      const modelJsonURL = window.URL.createObjectURL(
          new Blob([JSON.stringify(modelJSON)], {type: 'application/json'}));

      // If anchor elements are not provided, create them without attaching them
      // to parents, so that the downloaded file names can be controlled.
      const jsonAnchor = this.modelJsonAnchor == null ?
          document.createElement('a') :
          this.modelJsonAnchor;
      jsonAnchor.download = this.modelJsonFileName;
      jsonAnchor.href = modelJsonURL;
      // Trigger downloads by evoking a click event on the download anchors.
      // When multiple downloads are started synchronously, Firefox will only
      // save the last one.
      await defer(() => jsonAnchor.dispatchEvent(new MouseEvent('click')));

      if (modelArtifacts.weightData != null) {
        const weightDataAnchor = this.weightDataAnchor == null ?
            document.createElement('a') :
            this.weightDataAnchor;
        weightDataAnchor.download = this.weightDataFileName;
        weightDataAnchor.href = weightsURL;
        await defer(
            () => weightDataAnchor.dispatchEvent(new MouseEvent('click')));
      }

      return {modelArtifactsInfo: getModelArtifactsInfoForJSON(modelArtifacts)};
    }
  }
}

class BrowserFiles implements IOHandler {
  private readonly jsonFile: File;
  private readonly weightsFiles: File[];

  constructor(files: File[]) {
    if (files == null || files.length < 1) {
      throw new Error(
          `When calling browserFiles, at least 1 file is required, ` +
          `but received ${files}`);
    }
    this.jsonFile = files[0];
    this.weightsFiles = files.slice(1);
  }

  async load(): Promise<ModelArtifacts> {
    return new Promise((resolve, reject) => {
      const jsonReader = new FileReader();
      jsonReader.onload = (event: Event) => {
        let modelJSON: ModelJSON;
        try {
          // tslint:disable-next-line:no-any
          modelJSON = JSON.parse((event.target as any).result);
        } catch {
          reject(new Error(`Failed to parse file ${
            this.jsonFile.name}: {e.message}`));
          return;
        }

        const modelTopology = modelJSON.modelTopology;
        if (modelTopology == null) {
          reject(new Error(`modelTopology field is missing from file ${
              this.jsonFile.name}`));
          return;
        }

        const weightsManifest = modelJSON.weightsManifest;
        if (weightsManifest == null) {
          reject(new Error(`weightManifest field is missing from file ${
              this.jsonFile.name}`));
          return;
        }

        if (this.weightsFiles.length === 0) {
          resolve({modelTopology});
          return;
        }

        const modelArtifactsPromise = getModelArtifactsForJSON(
            modelJSON, (weightsManifest) => this.loadWeights(weightsManifest));
        resolve(modelArtifactsPromise);
      };

      jsonReader.onerror = error => reject(
          `Failed to read model topology and weights manifest JSON ` +
          `from file '${this.jsonFile.name}'. BrowserFiles supports loading ` +
          `Keras-style tf.Model artifacts only.`);
      jsonReader.readAsText(this.jsonFile);
    });
  }

  private loadWeights(weightsManifest: WeightsManifestConfig): Promise<[
    /* weightSpecs */ WeightsManifestEntry[], WeightData,
  ]> {
    const weightSpecs: WeightsManifestEntry[] = [];
    const paths: string[] = [];
    for (const entry of weightsManifest) {
      weightSpecs.push(...entry.weights);
      paths.push(...entry.paths);
    }

    const pathToFile: {[path: string]: File} =
        this.checkManifestAndWeightFiles(weightsManifest);

    const promises: Array<Promise<ArrayBuffer>> =
        paths.map(path => this.loadWeightsFile(path, pathToFile[path]));

    return Promise.all(promises).then(
        buffers => [weightSpecs, buffers]);
  }

  private loadWeightsFile(path: string, file: File): Promise<ArrayBuffer> {
    return new Promise((resolve, reject) => {
      const weightFileReader = new FileReader();
      weightFileReader.onload = (event: Event) => {
        // tslint:disable-next-line:no-any
        const weightData = (event.target as any).result as ArrayBuffer;
        resolve(weightData);
      };
      weightFileReader.onerror = error =>
          reject(`Failed to weights data from file of path '${path}'.`);
      weightFileReader.readAsArrayBuffer(file);
    });
  }

  /**
   * Check the compatibility between weights manifest and weight files.
   */
  private checkManifestAndWeightFiles(manifest: WeightsManifestConfig):
      {[path: string]: File} {
    const basenames: string[] = [];
    const fileNames = this.weightsFiles.map(file => basename(file.name));
    const pathToFile: {[path: string]: File} = {};
    for (const group of manifest) {
      group.paths.forEach(path => {
        const pathBasename = basename(path);
        if (basenames.indexOf(pathBasename) !== -1) {
          throw new Error(
              `Duplicate file basename found in weights manifest: ` +
              `'${pathBasename}'`);
        }
        basenames.push(pathBasename);
        if (fileNames.indexOf(pathBasename) === -1) {
          throw new Error(
              `Weight file with basename '${pathBasename}' is not provided.`);
        } else {
          pathToFile[path] = this.weightsFiles[fileNames.indexOf(pathBasename)];
        }
      });
    }

    if (basenames.length !== this.weightsFiles.length) {
      throw new Error(
          `Mismatch in the number of files in weights manifest ` +
          `(${basenames.length}) and the number of weight files provided ` +
          `(${this.weightsFiles.length}).`);
    }
    return pathToFile;
  }
}

export const browserDownloadsRouter: IORouter = (url: string|string[]) => {
  if (!env().getBool('IS_BROWSER')) {
    return null;
  } else {
    if (!Array.isArray(url) && url.startsWith(BrowserDownloads.URL_SCHEME)) {
      return browserDownloads(url.slice(BrowserDownloads.URL_SCHEME.length));
    } else {
      return null;
    }
  }
};
IORouterRegistry.registerSaveRouter(browserDownloadsRouter);

/**
 * Creates an IOHandler that triggers file downloads from the browser.
 *
 * The returned `IOHandler` instance can be used as model exporting methods such
 * as `tf.Model.save` and supports only saving.
 *
 * ```js
 * const model = tf.sequential();
 * model.add(tf.layers.dense(
 *     {units: 1, inputShape: [10], activation: 'sigmoid'}));
 * const saveResult = await model.save('downloads://mymodel');
 * // This will trigger downloading of two files:
 * //   'mymodel.json' and 'mymodel.weights.bin'.
 * console.log(saveResult);
 * ```
 *
 * @param fileNamePrefix Prefix name of the files to be downloaded. For use with
 *   `tf.Model`, `fileNamePrefix` should follow either of the following two
 *   formats:
 *   1. `null` or `undefined`, in which case the default file
 *      names will be used:
 *      - 'model.json' for the JSON file containing the model topology and
 *        weights manifest.
 *      - 'model.weights.bin' for the binary file containing the binary weight
 *        values.
 *   2. A single string or an Array of a single string, as the file name prefix.
 *      For example, if `'foo'` is provided, the downloaded JSON
 *      file and binary weights file will be named 'foo.json' and
 *      'foo.weights.bin', respectively.
 * @param config Additional configuration for triggering downloads.
 * @returns An instance of `BrowserDownloads` `IOHandler`.
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Loading',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
export function browserDownloads(fileNamePrefix = 'model'): IOHandler {
  return new BrowserDownloads(fileNamePrefix);
}

/**
 * Creates an IOHandler that loads model artifacts from user-selected files.
 *
 * This method can be used for loading from files such as user-selected files
 * in the browser.
 * When used in conjunction with `tf.loadLayersModel`, an instance of
 * `tf.LayersModel` (Keras-style) can be constructed from the loaded artifacts.
 *
 * ```js
 * // Note: This code snippet won't run properly without the actual file input
 * //   elements in the HTML DOM.
 *
 * // Suppose there are two HTML file input (`<input type="file" ...>`)
 * // elements.
 * const uploadJSONInput = document.getElementById('upload-json');
 * const uploadWeightsInput = document.getElementById('upload-weights');
 * const model = await tf.loadLayersModel(tf.io.browserFiles(
 *     [uploadJSONInput.files[0], uploadWeightsInput.files[0]]));
 * ```
 *
 * @param files `File`s to load from. Currently, this function supports only
 *   loading from files that contain Keras-style models (i.e., `tf.Model`s), for
 *   which an `Array` of `File`s is expected (in that order):
 *   - A JSON file containing the model topology and weight manifest.
 *   - Optionally, one or more binary files containing the binary weights.
 *     These files must have names that match the paths in the `weightsManifest`
 *     contained by the aforementioned JSON file, or errors will be thrown
 *     during loading. These weights files have the same format as the ones
 *     generated by `tensorflowjs_converter` that comes with the `tensorflowjs`
 *     Python PIP package. If no weights files are provided, only the model
 *     topology will be loaded from the JSON file above.
 * @returns An instance of `Files` `IOHandler`.
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Loading',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
export function browserFiles(files: File[]): IOHandler {
  return new BrowserFiles(files);
}
