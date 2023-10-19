/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {KernelConfig, tensor1d, Unique, UniqueAttrs, UniqueInputs} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const uniqueConfig: KernelConfig = {
  kernelName: Unique,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {x} = args.inputs as UniqueInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {axis = 0} = args.attrs as unknown as UniqueAttrs;

    const axisTensor = tensor1d([axis], 'int32');

    try {
      const opAttrs = [
        createTensorsTypeOpAttr('T', x.dtype),
        createTensorsTypeOpAttr('Taxis', 'int32'),
        createTensorsTypeOpAttr('out_idx', 'int32')
      ];
      const inputs = [x, axisTensor];
      return backend.executeMultipleOutputs('UniqueV2', opAttrs, inputs, 2);
    }
    finally {
      axisTensor.dispose();
    }
  }
};
