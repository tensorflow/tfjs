/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {Abs, AbsInputs, KernelConfig, KernelFunc, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function simpleAbsImpl(vals: TypedArray): Float32Array {
  const resultValues = new Float32Array(vals.length);
  for (let i = 0; i < vals.length; ++i) {
    resultValues[i] = Math.abs(vals[i]);
  }
  return resultValues;
}

export const abs = (args: {inputs: AbsInputs, backend: MathBackendCPU}) => {
  const {x} = args.inputs;
  const cpuBackend = args.backend;

  assertNotComplex(x, 'abs');

  let resultValues = new Float32Array(util.sizeFromShape(x.shape));
  const values = cpuBackend.data.get(x.dataId).values as TypedArray;
  resultValues = simpleAbsImpl(values);

  return cpuBackend.makeOutput(resultValues, x.shape, 'float32');
};

export const absConfig: KernelConfig = {
  kernelName: Abs,
  backendName: 'cpu',
  kernelFunc: abs as {} as KernelFunc,
};
