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

import {kernel_impls, KernelConfig, KernelFunc, NonMaxSuppressionV3, NonMaxSuppressionV3Attrs, NonMaxSuppressionV3Inputs, TypedArray} from '@tensorflow/tfjs-core';
import {WebGPUBackend} from '../backend_webgpu';

export function nonMaxSuppressionV3(args: {
  inputs: NonMaxSuppressionV3Inputs,
  backend: WebGPUBackend,
  attrs: NonMaxSuppressionV3Attrs
}) {
  console.warn(
      'tf.nonMaxSuppression() in webgpu locks the UI thread. ' +
      'Call tf.nonMaxSuppressionAsync() instead');

  const {inputs, backend, attrs} = args;
  const {boxes, scores} = inputs;
  const {maxOutputSize, iouThreshold, scoreThreshold} = attrs;

  const boxesVals = backend.readSync(boxes.dataId) as TypedArray;
  const scoresVals = backend.readSync(scores.dataId) as TypedArray;

  const {selectedIndices} = kernel_impls.nonMaxSuppressionV3Impl(
      boxesVals, scoresVals, maxOutputSize, iouThreshold, scoreThreshold);

  return backend.makeTensorInfo(
      [selectedIndices.length], 'int32', new Int32Array(selectedIndices));
}

export const nonMaxSuppressionV3Config: KernelConfig = {
  kernelName: NonMaxSuppressionV3,
  backendName: 'webgpu',
  kernelFunc: nonMaxSuppressionV3 as {} as KernelFunc
};
