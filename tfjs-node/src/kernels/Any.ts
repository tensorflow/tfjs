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

import {Any, AnyAttrs, AnyInputs, KernelConfig, tensor1d, util} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const anyConfig: KernelConfig = {
  kernelName: Any,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {x} = args.inputs as AnyInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {axis, keepDims} = args.attrs as {} as AnyAttrs;

    const origAxes = util.parseAxisParam(axis, x.shape);
    const opAttrs = [
      {name: 'keep_dims', type: backend.binding.TF_ATTR_BOOL, value: keepDims},
      createTensorsTypeOpAttr('Tidx', 'int32')
    ];
    const axesTensor = tensor1d(origAxes, 'int32');
    const res = backend.executeSingleOutput(Any, opAttrs, [x, axesTensor]);
    axesTensor.dispose();
    return res;
  }
};
