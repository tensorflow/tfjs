/**
 * @license
 * Copyright 2023 Google LLC.
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

import {Einsum, EinsumInputs, KernelConfig, Tensor} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const einsumConfig: KernelConfig = {
  kernelName: Einsum,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {inputs, attrs} = args;
    const tensors = inputs as unknown as EinsumInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {equation} = attrs;

    const opAttrs = [
      {name: 'N', type: backend.binding.TF_ATTR_INT, value: tensors.length},
      {name: 'equation', type: backend.binding.TF_ATTR_STRING, value: equation},
      createTensorsTypeOpAttr('T', tensors as Tensor[])
    ];

    const tensorArray = Array.from(tensors);
    return backend.executeSingleOutput(Einsum, opAttrs, tensorArray);
  }
};
