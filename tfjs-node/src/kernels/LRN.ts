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

import {KernelConfig, LRN, LRNAttrs, LRNInputs} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

// tslint:disable-next-line: variable-name
export const LRNConfig: KernelConfig = {
  kernelName: LRN,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {x} = args.inputs as LRNInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {depthRadius, bias, alpha, beta} = args.attrs as unknown as LRNAttrs;

    const opAttrs = [
      createTensorsTypeOpAttr('T', x.dtype),
      {
        name: 'depth_radius',
        type: backend.binding.TF_ATTR_INT,
        value: depthRadius
      },
      {name: 'bias', type: backend.binding.TF_ATTR_FLOAT, value: bias},
      {name: 'alpha', type: backend.binding.TF_ATTR_FLOAT, value: alpha},
      {name: 'beta', type: backend.binding.TF_ATTR_FLOAT, value: beta},
    ];
    return backend.executeSingleOutput(LRN, opAttrs, [x]);
  }
};
