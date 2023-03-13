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

import {KernelConfig, SpaceToBatchND, SpaceToBatchNDAttrs, SpaceToBatchNDInputs, tensor1d, tensor2d} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const spaceToBatchNDConfig: KernelConfig = {
  kernelName: SpaceToBatchND,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {x} = args.inputs as SpaceToBatchNDInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {blockShape, paddings} = args.attrs as unknown as SpaceToBatchNDAttrs;

    const blockShapeTensor = tensor1d(blockShape, 'int32');
    const paddingsTensor =
        tensor2d(paddings, [paddings.length, paddings[0].length], 'int32');
    const opAttrs = [
      createTensorsTypeOpAttr('T', x.dtype),
      createTensorsTypeOpAttr('Tblock_shape', 'int32'),
      createTensorsTypeOpAttr('Tpaddings', paddingsTensor.dtype)
    ];
    const res = backend.executeSingleOutput(
        SpaceToBatchND, opAttrs, [x, blockShapeTensor, paddingsTensor]);

    blockShapeTensor.dispose();
    paddingsTensor.dispose();

    return res;
  }
};
