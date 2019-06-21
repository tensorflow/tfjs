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
  constructor(
      private readonly modelTopology?: {}|ArrayBuffer,
      private readonly weightSpecs?: WeightsManifestEntry[],
      private readonly weightData?: ArrayBuffer,
      private readonly trainingConfig?: TrainingConfig) {}

  async load(): Promise<ModelArtifacts> {
    let result = {};
    if (this.modelTopology != null) {
      result = {modelTopology: this.modelTopology, ...result};
    }
    if (this.weightSpecs != null && this.weightSpecs.length > 0) {
      result = {weightSpecs: this.weightSpecs, ...result};
    }
    if (this.weightData != null && this.weightData.byteLength > 0) {
      result = {weightData: this.weightData, ...result};
    }
    if (this.trainingConfig != null) {
      result = {trainingConfig: this.trainingConfig, ...result};
    }
    return result;
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
 * @param modelTopology a object containing model topology (i.e., parsed from
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
    modelTopology: {}, weightSpecs?: WeightsManifestEntry[],
    weightData?: ArrayBuffer, trainingConfig?: TrainingConfig): IOHandler {
  // TODO(cais): The arguments should probably be consolidated into a single
  // object, with proper deprecation process. Even though this function isn't
  // documented, it is public and being used by some downstream libraries.
  return new PassthroughLoader(
      modelTopology, weightSpecs, weightData, trainingConfig);
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
