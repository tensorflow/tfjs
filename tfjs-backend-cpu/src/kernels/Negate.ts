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

import {KernelConfig, KernelFunc, Negate, TensorInfo, UnaryInputs, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';
import {multiply} from './Multiply';

export function negate(args: {inputs: UnaryInputs, backend: MathBackendCPU}):
    TensorInfo {
  const {inputs, backend} = args;
  const {x} = inputs;

  assertNotComplex(x, 'neg');

  const minusOne = util.createScalarValue(-1 as {} as 'float32', x.dtype);
  const minusOneTensor = backend.makeTensorInfo([], x.dtype, minusOne);
  const res =
      multiply({inputs: {a: minusOneTensor, b: x}, backend}) as TensorInfo;

  backend.disposeIntermediateTensorInfo(minusOneTensor);

  return res;
}

export const negateConfig: KernelConfig = {
  kernelName: Negate,
  backendName: 'cpu',
  kernelFunc: negate as {} as KernelFunc
};
