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

import {KernelConfig, PadV2, PadV2Attrs, PadV2Inputs, scalar, tensor2d} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const padV2Config: KernelConfig = {
  kernelName: PadV2,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {x} = args.inputs as PadV2Inputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {paddings, constantValue} = args.attrs as {} as PadV2Attrs;

    // Bind tensor values
    const paddingsTensor = tensor2d(paddings, [paddings.length, 2], 'int32');
    const constantTensor = scalar(constantValue, x.dtype);

    const opAttrs = [
      createTensorsTypeOpAttr('T', x.dtype),
      createTensorsTypeOpAttr('Tpaddings', paddingsTensor.dtype)
    ];

    const res = backend.executeSingleOutput(
        PadV2, opAttrs, [x, paddingsTensor, constantTensor]);

    paddingsTensor.dispose();
    constantTensor.dispose();

    return res;
  }
};
