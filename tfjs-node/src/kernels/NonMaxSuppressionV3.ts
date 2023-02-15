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

import {KernelConfig, NonMaxSuppressionV3, NonMaxSuppressionV3Attrs, NonMaxSuppressionV3Inputs, scalar} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const nonMaxSuppressionV3Config: KernelConfig = {
  kernelName: NonMaxSuppressionV3,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {boxes, scores} = args.inputs as NonMaxSuppressionV3Inputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {maxOutputSize, iouThreshold, scoreThreshold} =
        args.attrs as unknown as NonMaxSuppressionV3Attrs;

    const opAttrs = [createTensorsTypeOpAttr('T', boxes.dtype)];

    const maxOutputSizeTensor = scalar(maxOutputSize, 'int32');
    const iouThresholdTensor = scalar(iouThreshold);
    const scoreThresholdTensor = scalar(scoreThreshold);
    const res = backend.executeSingleOutput(NonMaxSuppressionV3, opAttrs, [
      boxes, scores, maxOutputSizeTensor, iouThresholdTensor,
      scoreThresholdTensor
    ]);
    maxOutputSizeTensor.dispose();
    iouThresholdTensor.dispose();
    scoreThresholdTensor.dispose();
    return res;
  }
};
