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

import {AddN, AddNInputs, buffer, KernelConfig, KernelFunc, Tensor, TensorInfo, TypedArray} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function addN(args: {inputs: AddNInputs, backend: MathBackendCPU}):
    TensorInfo {
  const {inputs, backend} = args;
  const tensors = inputs as Tensor[];

  assertNotComplex(inputs, 'addN');

  const vals =
      tensors.map(t => backend.data.get(t.dataId).values as TypedArray);
  const outBuf = buffer(tensors[0].shape, tensors[0].dtype as 'float32');
  const outVals = outBuf.values;
  for (let i = 0; i < tensors.length; i++) {
    const currVals = vals[i];
    for (let j = 0; j < outVals.length; j++) {
      outVals[j] += currVals[j];
    }
  }

  return backend.makeTensorInfo(outBuf.shape, outBuf.dtype, outBuf.values);
}

export const addNConfig: KernelConfig = {
  kernelName: AddN,
  backendName: 'cpu',
  kernelFunc: addN as {} as KernelFunc
};
