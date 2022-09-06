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
import {kernel_impls, KernelConfig, KernelFunc, NonMaxSuppressionV5, NonMaxSuppressionV5Attrs, NonMaxSuppressionV5Inputs, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
export type TypedArray = Float32Array|Int32Array|Uint8Array;

export function nonMaxSuppressionV5(args: {
  inputs: NonMaxSuppressionV5Inputs,
  backend: WebGPUBackend,
  attrs: NonMaxSuppressionV5Attrs
}): [TensorInfo, TensorInfo] {
  console.warn(
      'tf.nonMaxSuppression() in webgpu locks the UI thread. ' +
      'Call tf.nonMaxSuppressionAsync() instead');

  const {inputs, backend, attrs} = args;
  const {boxes, scores} = inputs;
  const {maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma} = attrs;

  const boxesVals = backend.readSync(boxes.dataId) as TypedArray;
  const scoresVals = backend.readSync(scores.dataId) as TypedArray;

  const maxOutputSizeVal = maxOutputSize;
  const iouThresholdVal = iouThreshold;
  const scoreThresholdVal = scoreThreshold;
  const softNmsSigmaVal = softNmsSigma;

  const {selectedIndices, selectedScores} =
      kernel_impls.nonMaxSuppressionV5Impl(
          boxesVals, scoresVals, maxOutputSizeVal, iouThresholdVal,
          scoreThresholdVal, softNmsSigmaVal);

  return [
    backend.makeTensorInfo(
        [selectedIndices.length], 'int32', new Int32Array(selectedIndices)),
    backend.makeTensorInfo(
        [selectedScores.length], 'float32', new Float32Array(selectedScores))
  ];
}

export const nonMaxSuppressionV5Config: KernelConfig = {
  kernelName: NonMaxSuppressionV5,
  backendName: 'webgpu',
  kernelFunc: nonMaxSuppressionV5 as {} as KernelFunc
};
