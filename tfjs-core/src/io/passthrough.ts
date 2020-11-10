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
 * IOHandlers that pass through the in-memory ModelArtifacts format.
 */

import {IOHandler, ModelArtifacts, SaveResult, TrainingConfig, WeightsManifestEntry} from './types';

class PassthroughLoader implements IOHandler {
  constructor(private readonly modelArtifacts?: ModelArtifacts) {}

  async load(): Promise<ModelArtifacts> {
    return this.modelArtifacts;
  }
}

class PassthroughSaver implements IOHandler {
  constructor(
      private readonly saveHandler:
          (artifacts: ModelArtifacts) => Promise<SaveResult>) {}

  async save(modelArtifacts: ModelArtifacts) {
    return this.saveHandler(modelArtifacts);
  }
}

/**
 * Creates an IOHandler that loads model artifacts from memory.
 *
 * When used in conjunction with `tf.loadLayersModel`, an instance of
 * `tf.LayersModel` (Keras-style) can be constructed from the loaded artifacts.
 *
 * ```js
 * const model = await tf.loadLayersModel(tf.io.fromMemory(
 *     modelTopology, weightSpecs, weightData));
 * ```
 *
 * @param modelArtifacts a object containing model topology (i.e., parsed from
 *   the JSON format).
 * @param weightSpecs An array of `WeightsManifestEntry` objects describing the
 *   names, shapes, types, and quantization of the weight data.
 * @param weightData A single `ArrayBuffer` containing the weight data,
 *   concatenated in the order described by the weightSpecs.
 * @param trainingConfig Model training configuration. Optional.
 *
 * @returns A passthrough `IOHandler` that simply loads the provided data.
 */
export function fromMemory(
    modelArtifacts: {}|ModelArtifacts, weightSpecs?: WeightsManifestEntry[],
    weightData?: ArrayBuffer, trainingConfig?: TrainingConfig): IOHandler {
  if (arguments.length === 1) {
    const isModelArtifacts =
        (modelArtifacts as ModelArtifacts).modelTopology != null ||
        (modelArtifacts as ModelArtifacts).weightSpecs != null;
    if (isModelArtifacts) {
      return new PassthroughLoader(modelArtifacts as ModelArtifacts);
    } else {
      // Legacy support: with only modelTopology.
      // TODO(cais): Remove this deprecated API.
      console.warn(
          'Please call tf.io.fromMemory() with only one argument. ' +
          'The argument should be of type ModelArtifacts. ' +
          'The multi-argument signature of tf.io.fromMemory() has been ' +
          'deprecated and will be removed in a future release.');
      return new PassthroughLoader({modelTopology: modelArtifacts as {}});
    }
  } else {
    // Legacy support.
    // TODO(cais): Remove this deprecated API.
    console.warn(
        'Please call tf.io.fromMemory() with only one argument. ' +
        'The argument should be of type ModelArtifacts. ' +
        'The multi-argument signature of tf.io.fromMemory() has been ' +
        'deprecated and will be removed in a future release.');
    return new PassthroughLoader({
      modelTopology: modelArtifacts as {},
      weightSpecs,
      weightData,
      trainingConfig
    });
  }
}

/**
 * Creates an IOHandler that passes saved model artifacts to a callback.
 *
 * ```js
 * function handleSave(artifacts) {
 *   // ... do something with the artifacts ...
 *   return {modelArtifactsInfo: {...}, ...};
 * }
 *
 * const saveResult = model.save(tf.io.withSaveHandler(handleSave));
 * ```
 *
 * @param saveHandler A function that accepts a `ModelArtifacts` and returns a
 *     `SaveResult`.
 */
export function withSaveHandler(
    saveHandler: (artifacts: ModelArtifacts) =>
        Promise<SaveResult>): IOHandler {
  return new PassthroughSaver(saveHandler);
}
