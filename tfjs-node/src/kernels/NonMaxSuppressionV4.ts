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

import {KernelConfig, NonMaxSuppressionV4, NonMaxSuppressionV4Attrs, NonMaxSuppressionV4Inputs, scalar, Tensor1D, Tensor2D} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

// TODO(nsthorat, dsmilkov): Remove dependency on tensors, use dataId.
export const nonMaxSuppressionV4Config: KernelConfig = {
  kernelName: NonMaxSuppressionV4,
  backendName: 'tensorflow',
  kernelFunc: ({inputs, backend, attrs}) => {
    const {boxes, scores} = inputs as NonMaxSuppressionV4Inputs;
    const {maxOutputSize, iouThreshold, scoreThreshold, padToMaxOutputSize} =
        attrs as unknown as NonMaxSuppressionV4Attrs;
    const maxOutputSizeTensor = scalar(maxOutputSize, 'int32');
    const iouThresholdTensor = scalar(iouThreshold, 'float32');
    const scoreThresholdTensor = scalar(scoreThreshold, 'float32');

    const nodeBackend = backend as NodeJSKernelBackend;

    const opAttrs = [
      createTensorsTypeOpAttr('T', boxes.dtype),
      createTensorsTypeOpAttr('T_threshold', 'float32'), {
        name: 'pad_to_max_output_size',
        type: nodeBackend.binding.TF_ATTR_BOOL,
        value: padToMaxOutputSize
      }
    ];

    const [selectedIndices, validOutputs] = nodeBackend.executeMultipleOutputs(
        'NonMaxSuppressionV4', opAttrs,
        [
          boxes as Tensor2D, scores as Tensor1D, maxOutputSizeTensor,
          iouThresholdTensor, scoreThresholdTensor
        ],
        2);

    maxOutputSizeTensor.dispose();
    iouThresholdTensor.dispose();
    scoreThresholdTensor.dispose();

    return [selectedIndices, validOutputs];
  }
};
