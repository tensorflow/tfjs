/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {NonMaxSuppressionV5, NonMaxSuppressionV5Attrs, NonMaxSuppressionV5Inputs} from '@tensorflow/tfjs-core';
import {KernelConfig, TypedArray} from '@tensorflow/tfjs-core';
import {kernel_impls} from '@tensorflow/tfjs-core';
const nonMaxSuppressionV5 = kernel_impls.nonMaxSuppressionV5;
import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export const nonMaxSuppressionV5Config: KernelConfig = {
  kernelName: NonMaxSuppressionV5,
  backendName: 'cpu',
  kernelFunc: ({inputs, backend, attrs}) => {
    const {boxes, scores} = inputs as NonMaxSuppressionV5Inputs;
    const {maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma} =
        attrs as unknown as NonMaxSuppressionV5Attrs;

    const cpuBackend = backend as MathBackendCPU;

    assertNotComplex(boxes, 'NonMaxSuppressionWithScore');

    const boxesVals = cpuBackend.data.get(boxes.dataId).values as TypedArray;
    const scoresVals = cpuBackend.data.get(scores.dataId).values as TypedArray;

    const maxOutputSizeVal = maxOutputSize;
    const iouThresholdVal = iouThreshold;
    const scoreThresholdVal = scoreThreshold;
    const softNmsSigmaVal = softNmsSigma;

    const {selectedIndices, selectedScores} = nonMaxSuppressionV5(
        boxesVals, scoresVals, maxOutputSizeVal, iouThresholdVal,
        scoreThresholdVal, softNmsSigmaVal);

    return [selectedIndices, selectedScores];
  }
};
