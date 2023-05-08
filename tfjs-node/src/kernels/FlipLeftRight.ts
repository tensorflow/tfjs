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

import {FlipLeftRight, FlipLeftRightInputs, KernelConfig, tensor1d, util} from '@tensorflow/tfjs-core';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const flipLeftRightConfig: KernelConfig = {
  kernelName: FlipLeftRight,
  backendName: 'tensorflow',
  kernelFunc({inputs, backend}) {
    const nodeBackend = backend as NodeJSKernelBackend;
    const {image} = inputs as FlipLeftRightInputs;
    const opAttrs = [
      createTensorsTypeOpAttr('Tidx', 'int32'),
      createTensorsTypeOpAttr('T', image.dtype),
    ];
    const axes = util.parseAxisParam([2], image.shape);
    const axisTensor = tensor1d(axes, 'int32');
    const res =
        nodeBackend.executeSingleOutput(
            'ReverseV2', opAttrs, [image, axisTensor]);
    axisTensor.dispose();
    return res;
  }
};
