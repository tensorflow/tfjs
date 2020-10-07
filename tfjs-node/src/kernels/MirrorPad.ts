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

import {KernelConfig, MirrorPad, MirrorPadAttrs, MirrorPadInputs, tensor2d} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const mirrorPadConfig: KernelConfig = {
  kernelName: MirrorPad,
  backendName: 'tensorflow',
  kernelFunc: ({inputs, backend, attrs}) => {
    const {x} = inputs as MirrorPadInputs;
    const {paddings, mode} = attrs as {} as MirrorPadAttrs;

    const nodeBackend = backend as NodeJSKernelBackend;

    const paddingsTensor = tensor2d(paddings, [paddings.length, 2], 'int32');

    const opAttrs = [
      createTensorsTypeOpAttr('T', x.dtype),
      createTensorsTypeOpAttr('Tpaddings', paddingsTensor.dtype), {
        name: 'mode',
        type: nodeBackend.binding.TF_ATTR_STRING,
        value: mode.toUpperCase()
      }
    ];

    const output = nodeBackend.executeSingleOutput(
        'MirrorPad', opAttrs, [x, paddingsTensor]);

    paddingsTensor.dispose();

    return output;
  }
};
