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

import {KernelConfig, KernelFunc, Pack, PackAttrs, PackInputs, Tensor, TensorInfo} from '@tensorflow/tfjs';
import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export function pack(
    args: {inputs: PackInputs, backend: NodeJSKernelBackend, attrs: PackAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {axis} = attrs;

  const opAttrs = [
    {name: 'N', type: backend.binding.TF_ATTR_INT, value: inputs.length},
    createTensorsTypeOpAttr('T', inputs as Tensor[]),
    {name: 'axis', type: backend.binding.TF_ATTR_INT, value: axis}
  ];

  const res = backend.executeSingleOutput(Pack, opAttrs, inputs);
  return res;
}

export const packConfig: KernelConfig = {
  kernelName: Pack,
  backendName: 'tensorflow',
  kernelFunc: pack as unknown as KernelFunc
};
