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

import {backend_util, BroadcastArgs, BroadcastArgsInputs, KernelConfig} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const broadcastArgsConfig: KernelConfig = {
  kernelName: BroadcastArgs,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {s0, s1} = args.inputs as BroadcastArgsInputs;
    const backend = args.backend as NodeJSKernelBackend;

    const opAttrs = [createTensorsTypeOpAttr(
        'T', backend_util.upcastType(s0.dtype, s1.dtype))];
    return backend.executeSingleOutput(BroadcastArgs, opAttrs, [s0, s1]);
  }
};
