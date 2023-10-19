/**
 * @license
 * Copyright 2023 Google LLC.
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
import * as tfc from '@tensorflow/tfjs-core';

export interface TensorDetail {
  data: number[];
  shape: number[];
  dtype: tfc.DataType;
}

export interface GraphModeGoldenData {
  /** The model name. */
  readonly name: string;
  /** Url for loading the model with `tf.loadGraphModel`. */
  readonly url: string;
  /** Whether model is to be loaded from TF Hub. */
  readonly fromTFHub?: boolean;
  /**
   * Golden tensor values for `model.predict`.
   */
  readonly inputs: TensorDetail|TensorDetail[]|Record<string, TensorDetail>;
  /**
   * The returned tensor values of `model.predict(this.inputs)`.
   */
  readonly outputDetails: TensorDetail|TensorDetail[]|
      Record<string, TensorDetail>;
  /**
   * All intermediate node tensor values after calling
   * `model.predict(this.inputs)`. The object keys are the node names.
   */
  readonly intermediateDetails: Record<string, TensorDetail>;
}
