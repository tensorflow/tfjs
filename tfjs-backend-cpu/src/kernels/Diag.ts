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

import {buffer, Diag, DiagInputs, KernelConfig, KernelFunc, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

export function diag(args: {inputs: DiagInputs, backend: MathBackendCPU}):
    TensorInfo {
  const {inputs, backend} = args;
  const {x} = inputs;

  const xSize = util.sizeFromShape(x.shape);

  const xVals = backend.data.get(x.dataId).values as TypedArray;
  const outBuf = buffer([xSize, xSize], x.dtype);
  const vals = outBuf.values;
  for (let i = 0; i < xVals.length; i++) {
    vals[i * xSize + i] = xVals[i];
  }

  const outShape = [...x.shape, ...x.shape];

  return backend.makeTensorInfo(outShape, outBuf.dtype, outBuf.values);
}

export const diagConfig: KernelConfig = {
  kernelName: Diag,
  backendName: 'cpu',
  kernelFunc: diag as {} as KernelFunc
};
