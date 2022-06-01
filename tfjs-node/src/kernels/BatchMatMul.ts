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

import {BatchMatMul, BatchMatMulAttrs, BatchMatMulInputs, KernelConfig} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const batchMatMulConfig: KernelConfig = {
  kernelName: BatchMatMul,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {a, b} = args.inputs as BatchMatMulInputs;
    const backend = args.backend as NodeJSKernelBackend;
    const {transposeA, transposeB} = args.attrs as {} as BatchMatMulAttrs;

    const opAttrs = [
      createTensorsTypeOpAttr('T', a.dtype),
      {name: 'adj_x', type: backend.binding.TF_ATTR_BOOL, value: transposeA},
      {name: 'adj_y', type: backend.binding.TF_ATTR_BOOL, value: transposeB}
    ];

    // libtensorflow's BatchMatMulV2 op performs the same behavior as other tfjs
    // backends' BatchMatMul (supports broadcasting), so a string literal is
    // used here to point to libtensorflow's BatchMatMulV2 op, instead of using
    // const `BatchMatMul` ('BatchMatMul') to resolve node-backend's special
    // mapping.
    return backend.executeSingleOutput('BatchMatMulV2', opAttrs, [a, b]);
  }
};
