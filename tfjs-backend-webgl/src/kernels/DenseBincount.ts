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

import {DenseBincount, DenseBincountAttrs, DenseBincountInputs, KernelConfig, KernelFunc, TensorInfo, TypedArray} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {bincountImplCPU, bincountReduceImplCPU} from '../kernel_utils/shared';

export function denseBincount(args: {
  inputs: DenseBincountInputs,
  backend: MathBackendWebGL,
  attrs: DenseBincountAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x, weights} = inputs;
  const {size, binaryOutput} = attrs;

  if (x.shape.length === 1) {
    const xVals = backend.readSync(x.dataId) as TypedArray;
    const weightsVals = backend.readSync(weights.dataId) as TypedArray;

    const outVals =
        bincountImplCPU(xVals, weightsVals, weights.dtype, weights.shape, size);

    return backend.makeTensorInfo([size], weights.dtype, outVals);
  } else if (x.shape.length === 2) {
    const xBuf = backend.bufferSync(x);
    const weightsBuf = backend.bufferSync(weights);

    const outBuf = bincountReduceImplCPU(xBuf, weightsBuf, size, binaryOutput);

    return backend.makeTensorInfo(outBuf.shape, weights.dtype, outBuf.values);
  }

  throw new Error(
      `Error in denseBincount: input must be at most rank 2, but got rank` +
      `${x.shape.length}.`);
}

export const denseBincountConfig: KernelConfig = {
  kernelName: DenseBincount,
  backendName: 'webgl',
  kernelFunc: denseBincount as {} as KernelFunc
};
