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

import {KernelConfig, KernelFunc, RaggedGather, RaggedGatherAttrs, RaggedGatherInputs, TensorInfo, TypedArray} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {raggedGatherImplCPU} from '../kernel_utils/shared';

export function raggedGather(args: {
  inputs: RaggedGatherInputs,
  backend: MathBackendWebGL,
  attrs: RaggedGatherAttrs
}): TensorInfo[] {
  const {inputs, backend, attrs} = args;
  const {paramsNestedSplits, paramsDenseValues, indices} = inputs;
  const {outputRaggedRank} = attrs;

  const $paramsNestedSplits =
      paramsNestedSplits.map(t => backend.readSync(t.dataId) as TypedArray);
  const $paramsNestedSplitsShapes = paramsNestedSplits.map(t => t.shape);
  const $paramsDenseValues =
      backend.readSync(paramsDenseValues.dataId) as TypedArray;
  const $indices = backend.readSync(indices.dataId) as TypedArray;

  const [outputNestedSplits, outputDenseValues, outputDenseValuesShape] =
      raggedGatherImplCPU(
          $paramsNestedSplits, $paramsNestedSplitsShapes, $paramsDenseValues,
          paramsDenseValues.shape, paramsDenseValues.dtype, $indices,
          indices.shape, outputRaggedRank);

  const outputNestedSplitsTensors = outputNestedSplits.map(
      (splits) => backend.makeTensorInfo([splits.length], 'int32', splits));

  const outputDenseValuesTensor = backend.makeTensorInfo(
      outputDenseValuesShape, paramsDenseValues.dtype, outputDenseValues);

  return outputNestedSplitsTensors.concat([outputDenseValuesTensor]);
}

export const raggedGatherConfig: KernelConfig = {
  kernelName: RaggedGather,
  backendName: 'webgl',
  kernelFunc: raggedGather as unknown as KernelFunc,
};
