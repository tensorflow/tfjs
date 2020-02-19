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

import {KernelConfig, registerKernel} from '@tensorflow/tfjs-core';

// Can only be done once a core release with these exports is done
// import {SquaredDifference, SquaredDifferenceInputs} from '@tf/tfjs-core'

import {createTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const squaredDifference_: KernelConfig = {
  kernelName: 'SquaredDifference',
  backendName: 'tensorflow',
  kernelFunc: ({inputs, backend}) => {
    const {$a, $b} = inputs;  // as SquaredDifferenceInputs;

    const opAttrs = [createTypeOpAttr('T', $a.dtype)];
    const nodeBackend = backend as NodeJSKernelBackend;

    return nodeBackend.executeSingleOutput(
        'SquaredDifference', opAttrs, [$a, $b]);
  }
};

registerKernel(squaredDifference_);
