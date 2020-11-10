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

import {KernelConfig, KernelFunc, Real, RealInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

export function real(args: {inputs: RealInputs, backend: MathBackendCPU}):
    TensorInfo {
  const {inputs, backend} = args;
  const {input} = inputs;

  const real = backend.data.get(input.dataId).complexTensorInfos.real;
  const realVal = backend.data.get(real.dataId).values;

  // When complex tensor is disposed, its underlying parts will be disposed too.
  // Make new tensor out of the real value of the complex. This makes sure the
  // value is still accessible even if complex tensor is disposed.
  return backend.makeTensorInfo(real.shape, real.dtype, realVal);
}

export const realConfig: KernelConfig = {
  kernelName: Real,
  backendName: 'cpu',
  kernelFunc: real as {} as KernelFunc
};
