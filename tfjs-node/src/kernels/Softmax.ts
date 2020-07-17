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

import {KernelConfig, NamedTensorInfoMap, TensorInfo} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

interface SoftmaxInputs extends NamedTensorInfoMap {
  logits: TensorInfo;
}

export const softmaxConfig: KernelConfig = {
  kernelName: 'Softmax',
  backendName: 'tensorflow',
  kernelFunc: ({inputs, backend}) => {
    const {logits} = inputs as SoftmaxInputs;
    const opAttrs = [createTensorsTypeOpAttr('T', logits.dtype)];

    const nodeBackend = backend as NodeJSKernelBackend;

    return nodeBackend.executeSingleOutput('Softmax', opAttrs, [logits]);
  }
};
