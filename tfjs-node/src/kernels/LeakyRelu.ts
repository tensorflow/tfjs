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

import {KernelConfig, LeakyRelu, LeakyReluAttrs, LeakyReluInputs, Tensor} from '@tensorflow/tfjs';
import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const leakyReluConfig: KernelConfig = {
  kernelName: LeakyRelu,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const inputs = args.inputs as LeakyReluInputs;
    const attrs = args.attrs as {} as LeakyReluAttrs;
    const backend = args.backend as NodeJSKernelBackend;
    const x = inputs.x as Tensor;
    const alpha = attrs.alpha;

    const opAttrs = [
      {name: 'alpha', type: backend.binding.TF_ATTR_FLOAT, value: alpha},
      createTensorsTypeOpAttr('T', x.dtype)
    ];

    return backend.executeSingleOutput(LeakyRelu, opAttrs, [x]);
  }
};
