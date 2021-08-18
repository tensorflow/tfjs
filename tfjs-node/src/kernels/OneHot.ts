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

import {KernelConfig, OneHot, OneHotAttrs, OneHotInputs, scalar} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const oneHotConfig: KernelConfig = {
  kernelName: OneHot,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {indices} = args.inputs as OneHotInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {depth, onValue, offValue} = args.attrs as {} as OneHotAttrs;

    const depthTensor = scalar(depth, 'int32');
    const onValueTensor = scalar(onValue, 'int32');
    const offValueTensor = scalar(offValue, 'int32');

    const opAttrs = [
      {name: 'axis', type: backend.binding.TF_ATTR_INT, value: -1},
      createTensorsTypeOpAttr('T', indices.dtype),
      createTensorsTypeOpAttr('TI', indices.dtype)
    ];

    const res = backend.executeSingleOutput(
        OneHot, opAttrs, [indices, depthTensor, onValueTensor, offValueTensor]);
    depthTensor.dispose();
    onValueTensor.dispose();
    offValueTensor.dispose();
    return res;
  }
};
