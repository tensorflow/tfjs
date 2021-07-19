/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {kernel_impls, KernelConfig, KernelFunc, NonMaxSuppressionV4, NonMaxSuppressionV4Attrs, NonMaxSuppressionV4Inputs, TensorInfo, TypedArray} from '@tensorflow/tfjs-core';

const nonMaxSuppressionV4Impl = kernel_impls.nonMaxSuppressionV4Impl;
import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function nonMaxSuppressionV4(args: {
  inputs: NonMaxSuppressionV4Inputs,
  backend: MathBackendCPU,
  attrs: NonMaxSuppressionV4Attrs
}): [TensorInfo, TensorInfo] {
  const {inputs, backend, attrs} = args;
  const {boxes, scores} = inputs;
  const {maxOutputSize, iouThreshold, scoreThreshold, padToMaxOutputSize} =
      attrs;

  assertNotComplex(boxes, 'NonMaxSuppressionPadded');

  const boxesVals = backend.data.get(boxes.dataId).values as TypedArray;
  const scoresVals = backend.data.get(scores.dataId).values as TypedArray;

  const {selectedIndices, validOutputs} = nonMaxSuppressionV4Impl(
      boxesVals, scoresVals, maxOutputSize, iouThreshold, scoreThreshold,
      padToMaxOutputSize);

  return [
    backend.makeTensorInfo(
        [selectedIndices.length], 'int32', new Int32Array(selectedIndices)),
    backend.makeTensorInfo([], 'int32', new Int32Array([validOutputs]))
  ];
}
export const nonMaxSuppressionV4Config: KernelConfig = {
  kernelName: NonMaxSuppressionV4,
  backendName: 'cpu',
  kernelFunc: nonMaxSuppressionV4 as {} as KernelFunc
};
