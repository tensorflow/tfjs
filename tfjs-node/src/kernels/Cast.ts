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

import {Cast, CastAttrs, CastInputs, KernelConfig} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const castConfig: KernelConfig = {
  kernelName: Cast,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {x} = args.inputs as CastInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {dtype} = args.attrs as {} as CastAttrs;

    const opAttrs = [
      createTensorsTypeOpAttr('SrcT', x.dtype),
      createTensorsTypeOpAttr('DstT', dtype),
      {name: 'Truncate', type: backend.binding.TF_ATTR_BOOL, value: false}
    ];
    return backend.executeSingleOutput(Cast, opAttrs, [x]);
  }
};
