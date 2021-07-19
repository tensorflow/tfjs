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

import {KernelConfig, KernelFunc, LeakyRelu, LeakyReluAttrs, LeakyReluInputs, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function leakyRelu(args: {
  inputs: LeakyReluInputs,
  backend: MathBackendCPU,
  attrs: LeakyReluAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {alpha} = attrs;

  assertNotComplex([x], 'leakyRelu');

  const xSize = util.sizeFromShape(x.shape);
  const xVals = backend.data.get(x.dataId).values as TypedArray;
  const outVals = util.getTypedArrayFromDType('float32', xSize);

  for (let i = 0; i < xVals.length; i++) {
    outVals[i] = xVals[i] < 0 ? alpha * xVals[i] : xVals[i];
  }

  return backend.makeTensorInfo(x.shape, 'float32', outVals);
}

export const leakyReluConfig: KernelConfig = {
  kernelName: LeakyRelu,
  backendName: 'cpu',
  kernelFunc: leakyRelu as {} as KernelFunc
};
