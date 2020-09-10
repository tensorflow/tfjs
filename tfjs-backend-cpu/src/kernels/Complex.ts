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

import {Complex, ComplexInputs, KernelConfig, KernelFunc, TensorInfo, TypedArray} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

export function complex(args: {inputs: ComplexInputs, backend: MathBackendCPU}):
    TensorInfo {
  const {inputs, backend} = args;
  const {real, imag} = inputs;

  const realVals = backend.data.get(real.dataId).values as TypedArray;
  const imagVals = backend.data.get(imag.dataId).values as TypedArray;

  const complexInfo = backend.makeTensorInfo(real.shape, 'complex64');

  const complex = backend.data.get(complexInfo.dataId);

  // The complex tensor owns the underlying real and imag tensorInfos, only the
  // complex tensor tracks refCount, when complexData is disposed the
  // underlying tensorData will be disposed.
  complex.complexTensorInfos = {
    real: backend.makeTensorInfo(real.shape, 'float32', realVals),
    imag: backend.makeTensorInfo(imag.shape, 'float32', imagVals)
  };

  return complexInfo;
}

export const complexConfig: KernelConfig = {
  kernelName: Complex,
  backendName: 'cpu',
  kernelFunc: complex as {} as KernelFunc
};
