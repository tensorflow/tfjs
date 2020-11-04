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

import {DataType, KernelConfig, KernelFunc, Negate, TensorInfo, TypedArray, UnaryInputs, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';
import {multiplyImpl} from './Multiply';

export function negateImpl(
    xVals: TypedArray, xShape: number[],
    xDtype: DataType): [TypedArray, number[]] {
  const minusOne =
      util.createScalarValue(-1 as {} as 'float32', xDtype) as TypedArray;
  return multiplyImpl([], xShape, minusOne, xVals, xDtype);
}

export function negate(args: {inputs: UnaryInputs, backend: MathBackendCPU}):
    TensorInfo {
  const {inputs, backend} = args;
  const {x} = inputs;

  assertNotComplex(x, 'neg');

  const xVals = backend.data.get(x.dataId).values as TypedArray;
  const [res, newShape] = negateImpl(xVals, x.shape, x.dtype);

  return backend.makeTensorInfo(newShape, x.dtype, res);
}

export const negateConfig: KernelConfig = {
  kernelName: Negate,
  backendName: 'cpu',
  kernelFunc: negate as {} as KernelFunc
};
