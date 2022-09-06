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

import {Concat, ConcatAttrs, ConcatInputs, KernelConfig, scalar, Tensor} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const concatConfig: KernelConfig = {
  kernelName: Concat,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const tensors = args.inputs as {} as ConcatInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {axis} = args.attrs as {} as ConcatAttrs;

    const opAttrs = [
      {name: 'N', type: backend.binding.TF_ATTR_INT, value: tensors.length}, {
        name: 'Tidx',
        type: backend.binding.TF_ATTR_TYPE,
        value: backend.binding.TF_INT32
      },
      createTensorsTypeOpAttr('T', tensors as Tensor[])
    ];

    const inputs = Array.from(tensors);
    const axisTensor = scalar(axis, 'int32');
    inputs.push(axisTensor);
    const res = backend.executeSingleOutput('ConcatV2', opAttrs, inputs);
    axisTensor.dispose();
    return res;
  }
};
