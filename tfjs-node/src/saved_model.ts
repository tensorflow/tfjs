/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

import {InferenceModel, ModelPredictConfig, NamedTensorMap, Tensor} from '@tensorflow/tfjs';
import {TensorInfo} from '@tensorflow/tfjs-converter/dist/src/data/types';
import {NodeJSKernelBackend} from './nodejs_kernel_backend';
import {ensureTensorflowBackend, nodeBackend} from './ops/op_utils';

export class TFSavedModel implements InferenceModel {
  private readonly id: number;
  private readonly backend: NodeJSKernelBackend;
  private deleted: boolean;

  constructor(id: number, backend: NodeJSKernelBackend) {
    this.id = id;
    this.backend = backend;
    this.deleted = false;
  }

  /** Placeholder function. */
  get inputs(): TensorInfo[] {
    throw new Error('SavedModel inputs information is not available yet.');
  }

  /** Placeholder function. */
  get outputs(): TensorInfo[] {
    throw new Error('SavedModel outputs information is not available yet.');
  }

  /**
   * Delete the SavedModel from nodeBackend and delete corresponding object in
   * the C++ backend.
   */
  delete() {
    if (!this.deleted) {
      this.deleted = true;
      this.backend.deleteSavedModel(this.id);
    } else {
      throw new Error('This SavedModel has been deleted.');
    }
  }

  /**
   * Placeholder function.
   * @param inputs
   * @param config
   */
  predict(inputs: Tensor|Tensor[]|NamedTensorMap, config?: ModelPredictConfig):
      Tensor|Tensor[]|NamedTensorMap {
    throw new Error(
        'predict() function of TFSavedModel is not implemented yet.');
  }

  /**
   * Placeholder function.
   * @param inputs
   * @param outputs
   */
  execute(inputs: Tensor|Tensor[]|NamedTensorMap, outputs: string|string[]):
      Tensor|Tensor[] {
    throw new Error(
        'Execute() function of TFSavedModel is not implemented yet.');
  };
}

/**
 * Decode a JPEG-encoded image to a 3D Tensor of dtype `int32`.
 *
 * @param path The path of the exported SavedModel
 */
/**
 * @doc {heading: 'SavedModel', namespace: 'node'}
 */
export async function loadSavedModel(path: string): Promise<TFSavedModel> {
  ensureTensorflowBackend();
  return nodeBackend().loadSavedModel(path);
}
