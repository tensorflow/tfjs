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

import {Imag, ImagInputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

export function imag(args: {inputs: ImagInputs, backend: MathBackendCPU}):
    TensorInfo {
  const {inputs, backend} = args;
  const {input} = inputs;

  const imag = backend.data.get(input.dataId).complexTensorInfos.imag;
  const imagVal = backend.data.get(imag.dataId).values;

  // When complex tensor is disposed, its underlying parts will be disposed too.
  // Make new tensor out of the imag value of the complex. This makes sure the
  // value is still accessible even if complex tensor is disposed.
  return backend.makeTensorInfo(imag.shape, imag.dtype, imagVal);
}

export const imagConfig: KernelConfig = {
  kernelName: Imag,
  backendName: 'cpu',
  kernelFunc: imag as {} as KernelFunc
};
