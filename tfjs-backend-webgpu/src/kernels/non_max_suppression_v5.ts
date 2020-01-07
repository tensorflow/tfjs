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

import {backend_util, NamedAttrMap, NamedTensorInfoMap, registerKernel, TensorInfo} from '@tensorflow/tfjs-core';
import {WebGPUBackend} from '../backend_webgpu';

interface NonMaxSuppressionWithScoreInputs extends NamedTensorInfoMap {
  boxes: TensorInfo;
  scores: TensorInfo;
}

interface NonMaxSuppressionWithScoreAttrs extends NamedAttrMap {
  maxOutputSize: number;
  iouThreshold: number;
  scoreThreshold: number;
  softNmsSigma: number;
}

registerKernel({
  kernelName: 'NonMaxSuppressionV5',
  backendName: 'webgpu',
  kernelFunc: ({inputs, backend, attrs}) => {
    console.warn(
        'tf.nonMaxSuppression() in webgpu locks the UI thread. ' +
        'Call tf.nonMaxSuppressionAsync() instead');

    const {boxes, scores} = inputs as NonMaxSuppressionWithScoreInputs;
    const {maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma} =
        attrs as NonMaxSuppressionWithScoreAttrs;

    const gpuBackend = backend as WebGPUBackend;

    const boxesVals =
        gpuBackend.readSync(boxes.dataId) as backend_util.TypedArray;
    const scoresVals =
        gpuBackend.readSync(scores.dataId) as backend_util.TypedArray;

    const maxOutputSizeVal = maxOutputSize;
    const iouThresholdVal = iouThreshold;
    const scoreThresholdVal = scoreThreshold;
    const softNmsSigmaVal = softNmsSigma;

    const {selectedIndices, selectedScores} = backend_util.nonMaxSuppressionV5(
        boxesVals, scoresVals, maxOutputSizeVal, iouThresholdVal,
        scoreThresholdVal, softNmsSigmaVal);

    return [selectedIndices, selectedScores];
  }
});
