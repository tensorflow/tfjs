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

import {Int, IntInputs, KernelConfig, KernelFunc, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function int(args: {inputs: IntInputs, backend: MathBackendCPU}):
    TensorInfo {
  const {inputs, backend} = args;
  const {x} = inputs;

  assertNotComplex(x, 'int');

  const values = backend.data.get(x.dataId).values as TypedArray;

  const resultValues = new Int32Array(util.sizeFromShape(x.shape));

  for (let i = 0; i < values.length; ++i) {
    resultValues[i] = values[i];
  }

  return backend.makeTensorInfo(resultValues, x.shape, 'int32');
}

export const intConfig: KernelConfig = {
  kernelName: Int,
  backendName: 'cpu',
  kernelFunc: int as {} as KernelFunc
};
