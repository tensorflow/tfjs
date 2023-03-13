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

import {KernelConfig, KernelFunc, RaggedTensorToTensor, RaggedTensorToTensorAttrs, RaggedTensorToTensorInputs, TensorInfo, TypedArray} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

import {raggedTensorToTensorImpl} from './RaggedTensorToTensor_impl';

export function raggedTensorToTensor(args: {
  inputs: RaggedTensorToTensorInputs,
  backend: MathBackendCPU,
  attrs: RaggedTensorToTensorAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {shape, values, defaultValue, rowPartitionTensors} = inputs;
  const {rowPartitionTypes} = attrs;

  const $shape = backend.data.get(shape.dataId).values as TypedArray;
  const $values = backend.data.get(values.dataId).values as TypedArray;
  const $defaultValue =
      backend.data.get(defaultValue.dataId).values as TypedArray;
  const $rowPartitionValues = rowPartitionTensors.map(
      t => backend.data.get(t.dataId).values as TypedArray);
  const rowPartitionValuesShapes = rowPartitionTensors.map(t => t.shape);

  const [outputShape, output] = raggedTensorToTensorImpl(
      $shape, shape.shape, $values, values.shape, values.dtype, $defaultValue,
      defaultValue.shape, $rowPartitionValues, rowPartitionValuesShapes,
      rowPartitionTypes);
  return backend.makeTensorInfo(outputShape, values.dtype, output);
}

export const raggedTensorToTensorConfig: KernelConfig = {
  kernelName: RaggedTensorToTensor,
  backendName: 'cpu',
  kernelFunc: raggedTensorToTensor as unknown as KernelFunc,
};
