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

import {ArgMax, ArgMaxAttrs, ArgMaxInputs, KernelConfig, scalar, Tensor} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const argMaxConfig: KernelConfig = {
  kernelName: ArgMax,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {x} = args.inputs as ArgMaxInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {axis} = args.attrs as {} as ArgMaxAttrs;

    const toDispose: Tensor[] = [];
    let xInput = x;
    if (x.dtype === 'bool') {
      xInput = (x as Tensor).toInt();
      toDispose.push(xInput as Tensor);
    }
    const axisScalar = scalar(axis, 'int32');
    toDispose.push(axisScalar);
    const opAttrs = [
      createTensorsTypeOpAttr('T', xInput.dtype),
      createTensorsTypeOpAttr('Tidx', 'int32'),
      createTensorsTypeOpAttr('output_type', 'int32')
    ];

    const res =
        backend.executeSingleOutput(ArgMax, opAttrs, [xInput, axisScalar]);
    toDispose.forEach(t => t.dispose());
    return res;
  }
};
