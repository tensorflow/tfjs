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

import {KernelConfig} from '@tensorflow/tfjs';
import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const squaredDifferenceConfig: KernelConfig = {
  // TODO import this kernelName from core once exported.
  kernelName: 'SquaredDifference',
  backendName: 'tensorflow',
  kernelFunc: ({inputs, backend}) => {
    const {a, b} = inputs;

    const opAttrs = [createTensorsTypeOpAttr('T', a.dtype)];
    const nodeBackend = backend as NodeJSKernelBackend;

    return nodeBackend.executeSingleOutput(
        'SquaredDifference', opAttrs, [a, b]);
  }
};
