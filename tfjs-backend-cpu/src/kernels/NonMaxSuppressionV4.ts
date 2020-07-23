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

import {NonMaxSuppressionV4, NonMaxSuppressionV4Attrs, NonMaxSuppressionV4Inputs} from '@tensorflow/tfjs-core';
import {KernelConfig, TypedArray} from '@tensorflow/tfjs-core';
import {kernel_impls} from '@tensorflow/tfjs-core';
const nonMaxSuppressionV4Impl = kernel_impls.nonMaxSuppressionV4Impl;
import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export const nonMaxSuppressionV4Config: KernelConfig = {
  kernelName: NonMaxSuppressionV4,
  backendName: 'cpu',
  kernelFunc: ({inputs, backend, attrs}) => {
    const {boxes, scores} = inputs as NonMaxSuppressionV4Inputs;
    const {maxOutputSize, iouThreshold, scoreThreshold, padToMaxOutputSize} =
        attrs as unknown as NonMaxSuppressionV4Attrs;

    const cpuBackend = backend as MathBackendCPU;

    assertNotComplex(boxes, 'NonMaxSuppressionPadded');

    const boxesVals = cpuBackend.data.get(boxes.dataId).values as TypedArray;
    const scoresVals = cpuBackend.data.get(scores.dataId).values as TypedArray;

    const {selectedIndices, validOutputs} = nonMaxSuppressionV4Impl(
        boxesVals, scoresVals, maxOutputSize, iouThreshold, scoreThreshold,
        padToMaxOutputSize);

    return [selectedIndices, validOutputs];
  }
};
