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

import {KernelConfig, Reverse, ReverseAttrs, ReverseInputs, tensor1d, util} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const reverseConfig: KernelConfig = {
  kernelName: Reverse,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {x} = args.inputs as ReverseInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {dims} = args.attrs as unknown as ReverseAttrs;

    const opAttrs = [
      createTensorsTypeOpAttr('Tidx', 'int32'),
      createTensorsTypeOpAttr('T', x.dtype)
    ];

    const axes = util.parseAxisParam(dims, x.shape);
    const axisTensor = tensor1d(axes, 'int32');
    const res =
        backend.executeSingleOutput('ReverseV2', opAttrs, [x, axisTensor]);
    axisTensor.dispose();
    return res;
  }
};
