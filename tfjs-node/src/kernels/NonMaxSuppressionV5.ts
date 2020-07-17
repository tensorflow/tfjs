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

import {KernelConfig, NamedAttrMap, NamedTensorInfoMap, scalar, Tensor1D, Tensor2D, TensorInfo} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

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

// TODO(nsthorat, dsmilkov): Remove dependency on tensors, use dataId.
export const nonMaxSuppressionV5Config: KernelConfig = {
  kernelName: 'NonMaxSuppressionV5',
  backendName: 'tensorflow',
  kernelFunc: ({inputs, backend, attrs}) => {
    const {boxes, scores} = inputs as NonMaxSuppressionWithScoreInputs;
    const {maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma} =
        attrs as NonMaxSuppressionWithScoreAttrs;
    const maxOutputSizeTensor = scalar(maxOutputSize, 'int32');
    const iouThresholdTensor = scalar(iouThreshold);
    const scoreThresholdTensor = scalar(scoreThreshold);
    const softNmsSigmaTensor = scalar(softNmsSigma);
    const opAttrs = [createTensorsTypeOpAttr('T', boxes.dtype)];

    const nodeBackend = backend as NodeJSKernelBackend;

    const [selectedIndices, selectedScores, validOutputs] =
        nodeBackend.executeMultipleOutputs(
            'NonMaxSuppressionV5', opAttrs,
            [
              boxes as Tensor2D, scores as Tensor1D, maxOutputSizeTensor,
              iouThresholdTensor, scoreThresholdTensor, softNmsSigmaTensor
            ],
            3);

    maxOutputSizeTensor.dispose();
    iouThresholdTensor.dispose();
    scoreThresholdTensor.dispose();
    softNmsSigmaTensor.dispose();
    validOutputs.dispose();

    return [selectedIndices, selectedScores];
  }
};
