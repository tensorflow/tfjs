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

import {backend_util, cast, KernelConfig, Pow, PowInputs, Tensor, tidy} from '@tensorflow/tfjs';

import {createTensorsTypeOpAttr, NodeJSKernelBackend} from '../nodejs_kernel_backend';

export const powConfig: KernelConfig = {
  kernelName: Pow,
  backendName: 'tensorflow',
  kernelFunc: (args) => {
    const {a, b} = args.inputs as PowInputs;
    const backend = args.backend as NodeJSKernelBackend;

    const dtype = backend_util.upcastType(a.dtype, b.dtype);
    const opAttrs = [createTensorsTypeOpAttr('T', dtype)];
    return tidy(() => {
      return backend.executeSingleOutput(
          Pow, opAttrs, [cast(a as Tensor, dtype), cast(b as Tensor, dtype)]);
    });
  }
};
