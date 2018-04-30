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

/* Type definitions for exporting and importing of models. */

/**
 * A map from Tensor dtype to number of bytes per element of the Tensor.
 */
export const DTYPE_VALUE_SIZE_MAP: {[dtype: string]: number} = {
  'float32': 4,
  'int32': 4,
  'uint16': 2,
  'uint8': 1,
  'bool': 1,
};

/**
 * A weight manifest.
 *
 * The weight manifest consists of an ordered list of weight-manifest groups.
 * Each weight-manifest group ("group" for short hereafter) consists of a
 * number of weight values stored in a number of paths.
 * See the documentation of `WeightManifestGroupConfig` below for more details.
 */
export type WeightsManifestConfig = WeightsManifestGroupConfig[];

/**
 * A weight-manifest group.
 *
 * Consists of an ordered list of weight values encoded in binary format,
 * stored in an ordered list of paths.
 */
export interface WeightsManifestGroupConfig {
  /**
   * An ordered list of paths.
   *
   * Paths are intentionally abstract in order to be general. For example, they
   * can be relative URL paths or relative paths on the file system.
   */
  paths: string[];

  /**
   * Specifications of the weights stored in the paths.
   */
  weights: WeightsManifestEntry[];
}

/**
 * An entry in the weight manifest.
 *
 * The entry contains specification of a weight.
 */
export interface WeightsManifestEntry {
  /**
   * Name of the weight, e.g., 'Dense_1/bias'
   */
  name: string;

  /**
   * Shape of the weight.
   */
  shape: number[];

  /**
   * Data type of the weight.
   */
  dtype: 'float32'|'int32'|'bool';

  /**
   * Information for dequantization of the weight.
   */
  quantization?: {
    scale: number,           // The scaling constant to multiply by.
    min: number,             // The (possibly nudged) minimum weight to add.
    dtype: 'uint16'|'uint8'  // The dtype of the quantized weights.
  };
}

/**
 * Options for saving a model.
 */
export interface SaveConfig {
  /**
   * Whether to save only the trainable weights of the model, ignoring the
   * untrainable ones.
   */
  trainableOnly?: boolean;
}

/**
 * Result of a saving operation.
 */
export interface SaveResult {
  /**
   * Information about the model artifacts saved.
   */
  modelArtifactsInfo: ModelArtifactsInfo;

  /**
   * HTTP responses from the server that handled the model-saving request (if
   * any). This is applicable only to server-based saving routes.
   */
  responses?: Response[];

  /**
   * Error messages and related data (if any).
   */
  errors?: Array<{}|string>;
}

export interface ModelArtifactsInfo {
  /**
   * Timestamp for when the model is saved.
   */
  dateSaved: Date;

  /**
   * Type of the model topology
   *
   * Possible values:
   *   - KerasJSON: Keras-style JSON config (human-readable string).
   *   - GraphDef: TensorFlow
   *     [GraphDef](https://www.tensorflow.org/extend/tool_developers/#graphdef)
   *     protocal buffer (binary).
   */
  modelTopologyType: 'KerasJSON'|'GraphDef';

  /**
   * Size of model topology (Keras JSON or GraphDef), in bytes.
   */
  modelTopologyBytes?: number;

  /**
   * Size of weight specification or manifest, in bytes.
   */
  weightSpecsBytes?: number;

  /**
   * Size of weight value data, in bytes.
   */
  weightDataBytes?: number;
}

/**
 * The serialized artifacts of a model, including topology and weights.
 *
 * The `modelTopology`, `weightSpecs` and `weightData` fields of this interface
 * are optional, in order to support topology- or weights-only saving and
 * loading.
 */
export interface ModelArtifacts {
  /**
   * Model topology.
   *
   * For Keras-style `tf.Model`s, this is a JSON object.
   * For TensorFlow-style models (e.g., `FrozenModel`), this is a binary buffer
   * carrying the `GraphDef` protocol buffer.
   */
  modelTopology?: {}|ArrayBuffer;

  /**
   * Weight specifications.
   *
   * This corresponds to the weightsData below.
   */
  weightSpecs?: WeightsManifestEntry[];

  /**
   * Binary buffer for all weight values concatenated in the order specified
   * by `weightSpecs`.
   */
  weightData?: ArrayBuffer;
}

/**
 * Type definition for handlers of loading operations.
 */
export type LoadHandler = () => Promise<ModelArtifacts>;

/**
 * Type definition for handlers of saving operations.
 */
export type SaveHandler = (modelArtifact: ModelArtifacts) =>
    Promise<SaveResult>;

/**
 * Interface for a model import/export handler.
 *
 * The `save` and `load` handlers are both optional, in order to allow handlers
 * that support only saving or loading.
 */
// tslint:disable-next-line:interface-name
export interface IOHandler {
  save?: SaveHandler;
  load?: LoadHandler;
}
