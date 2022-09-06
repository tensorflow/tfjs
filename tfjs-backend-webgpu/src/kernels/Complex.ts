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

import {Complex, ComplexInputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {identity} from './Identity';

/**
 * Complex tensors share data with their real and imaginary components. Complex
 * tensors' reference to the components is tracked by refCount on the individual
 * component. The refCounts are increased by the identity call.
 *
 * When a complex tensor is disposed, it will reduce the refCount on the
 * components by calling disposeData on each.
 */
export function complex(args: {inputs: ComplexInputs, backend: WebGPUBackend}):
    TensorInfo {
  const {inputs, backend} = args;
  const {real, imag} = inputs;

  const complexInfo = backend.makeTensorInfo(real.shape, 'complex64');
  const complex = backend.tensorMap.get(complexInfo.dataId);

  const realTensorInfo = identity({inputs: {x: real}, backend});

  const imagTensorInfo = identity({inputs: {x: imag}, backend});

  complex.complexTensorInfos = {real: realTensorInfo, imag: imagTensorInfo};

  return complexInfo;
}

export const complexConfig: KernelConfig = {
  kernelName: Complex,
  backendName: 'webgpu',
  kernelFunc: complex as {} as KernelFunc
};
