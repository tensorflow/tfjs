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

import {backend_util, kernel_impls, KernelConfig, NonMaxSuppressionV3, NonMaxSuppressionV3Attrs, NonMaxSuppressionV3Inputs, TypedArray} from '@tensorflow/tfjs-core';
import {MathBackendWebGL} from '../backend_webgl';

export const nonMaxSuppressionV3Config: KernelConfig = {
  kernelName: NonMaxSuppressionV3,
  backendName: 'webgl',
  kernelFunc: ({inputs, backend, attrs}) => {
    backend_util.warn(
        'tf.nonMaxSuppression() in webgl locks the UI thread. ' +
        'Call tf.nonMaxSuppressionAsync() instead');

    const {boxes, scores} = inputs as NonMaxSuppressionV3Inputs;
    const {maxOutputSize, iouThreshold, scoreThreshold} =
        attrs as unknown as NonMaxSuppressionV3Attrs;

    const gpuBackend = backend as MathBackendWebGL;

    const boxesVals = gpuBackend.readSync(boxes.dataId) as TypedArray;
    const scoresVals = gpuBackend.readSync(scores.dataId) as TypedArray;

    const maxOutputSizeVal = maxOutputSize;
    const iouThresholdVal = iouThreshold;
    const scoreThresholdVal = scoreThreshold;

    return kernel_impls.nonMaxSuppressionV3(
        boxesVals, scoresVals, maxOutputSizeVal, iouThresholdVal,
        scoreThresholdVal);
  }
};
