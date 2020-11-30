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
import {createSimpleBinaryKernelImpl} from '../utils/binary_impl';

const leakyReluImpl = createSimpleBinaryKernelImpl(
    (xValue: number, aValue: number) => xValue < 0 ? aValue * xValue : xValue);

export function leakyRelu(args: {
  inputs: LeakyReluInputs,
  backend: MathBackendCPU,
  attrs: LeakyReluAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {alpha} = attrs;

  assertNotComplex([x], 'leakyRelu');

  const $alpha = backend.makeTensorInfo(
      [], 'float32',
      util.createScalarValue(alpha as {} as 'float32', 'float32'));

  const xVals = backend.data.get(x.dataId).values as TypedArray;
  const $alphaVals = backend.data.get($alpha.dataId).values as TypedArray;
  const [resultData, resultShape] =
      leakyReluImpl(x.shape, $alpha.shape, xVals, $alphaVals, x.dtype);

  backend.disposeIntermediateTensorInfo($alpha);

  return backend.makeTensorInfo(resultShape, x.dtype, resultData);
}

export const leakyReluConfig: KernelConfig = {
  kernelName: LeakyRelu,
  backendName: 'cpu',
  kernelFunc: leakyRelu as {} as KernelFunc
};
