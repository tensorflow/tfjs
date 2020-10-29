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

import {KernelConfig, Unpack, UnpackAttrs, UnpackInputs} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const unpackConfig: KernelConfig = {
  kernelName: Unpack,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {value} = args.inputs as UnpackInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {axis} = args.attrs as {} as UnpackAttrs;

    if (axis >= value.shape.length) {
      throw new Error(
          `Invalid axis supplied: ${axis} shape length: ${value.shape.length}`);
    }
    const num = value.shape[axis];
    const opAttrs = [
      {name: 'num', type: backend.binding.TF_ATTR_INT, value: num},
      createTensorsTypeOpAttr('T', value.dtype),
      {name: 'axis', type: backend.binding.TF_ATTR_INT, value: axis}
    ];
    return backend.executeMultipleOutputs(Unpack, opAttrs, [value], num);
  }
};
