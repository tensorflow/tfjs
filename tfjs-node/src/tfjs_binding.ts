/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {backend_util} from '@tensorflow/tfjs-core';

export declare class TensorMetadata {
  id: number;
  shape: number[];
  dtype: number;
}

export declare class TFEOpAttr {
  name: string;
  type: number;
  value: boolean|number|object|string|number[];
}

export interface TFJSBinding {
  TensorMetadata: typeof TensorMetadata;
  TFEOpAttr: typeof TFEOpAttr;

  // Creates a tensor with the backend.
  createTensor(
      shape: number[], dtype: number,
      buffer: backend_util.BackendValues): number;

  // Deletes a tensor with the backend.
  deleteTensor(tensorId: number): void;

  // Reads data-sync from a tensor on the backend.
  tensorDataSync(tensorId: number): Float32Array|Int32Array|Uint8Array;

  // Executes an Op on the backend, returns an array of output TensorMetadata.
  executeOp(
      opName: string, opAttrs: TFEOpAttr[], inputTensorIds: number[],
      numOutputs: number): TensorMetadata[];

  // Load a SavedModel from a path.
  loadSavedModel(exportDir: string, tags: string): number;

  // Remove a SavedModel from memory.
  deleteSavedModel(savedModelId: number): void;

  // Execute a SavedModel with input, returns an array of output TensorMetadata.
  runSavedModel(
      savedModelId: number, inputTensorIds: number[], inputOpNames: string,
      outputOpNames: string): TensorMetadata[];

  getNumOfSavedModels(): number;

  isUsingGpuDevice(): boolean;

  // TF Types
  TF_FLOAT: number;
  TF_INT32: number;
  TF_INT64: number;
  TF_BOOL: number;
  TF_COMPLEX64: number;
  TF_STRING: number;
  TF_RESOURCE: number;
  TF_UINT8: number;

  // TF OpAttrTypes
  TF_ATTR_STRING: number;
  TF_ATTR_INT: number;
  TF_ATTR_FLOAT: number;
  TF_ATTR_BOOL: number;
  TF_ATTR_TYPE: number;
  TF_ATTR_SHAPE: number;
  TF_ATTR_RESOURCE: number;

  TF_Version: string;
}
