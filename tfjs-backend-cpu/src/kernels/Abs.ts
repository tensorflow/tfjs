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

export function simpleAbsImpl(vals: TypedArray): Float32Array {
  const resultValues = new Float32Array(vals.length);
  for (let i = 0; i < vals.length; ++i) {
    resultValues[i] = Math.abs(vals[i]);
  }
  return resultValues;
}

export const absKernelFunc =
    (args: {inputs: AbsInputs, backend: MathBackendCPU}) => {
      const {x} = args.inputs;
      const cpuBackend = args.backend;
      let resultValues = new Float32Array(util.sizeFromShape(x.shape));
      if (x.dtype !== 'complex64') {
        const values = cpuBackend.data.get(x.dataId).values as TypedArray;
        resultValues = simpleAbsImpl(values);
      } else {
        const complexVals = cpuBackend.data.get(x.dataId);
        const real = complexVals.complexTensorInfos.real;
        const imag = complexVals.complexTensorInfos.imag;
        const realVals =
            cpuBackend.data.get(real.dataId).values as Float32Array;
        const imagVals =
            cpuBackend.data.get(imag.dataId).values as Float32Array;
        for (let i = 0; i < realVals.length; i++) {
          const real = realVals[i];
          const imag = imagVals[i];
          resultValues[i] = Math.hypot(real, imag);
        }
      }
      return cpuBackend.makeOutput(resultValues, x.shape, 'float32');
    };

export const absConfig: KernelConfig = {
  kernelName: Abs,
  backendName: 'cpu',
  kernelFunc: absKernelFunc as {} as KernelFunc,
};
