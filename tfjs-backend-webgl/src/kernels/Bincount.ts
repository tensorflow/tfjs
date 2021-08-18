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

import {Bincount, BincountAttrs, BincountInputs, KernelConfig, KernelFunc, TensorInfo, TypedArray} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {bincountImplCPU} from '../kernel_utils/shared';

export function bincount(args: {
  inputs: BincountInputs,
  backend: MathBackendWebGL,
  attrs: BincountAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x, weights} = inputs;
  const {size} = attrs;

  const xVals = backend.readSync(x.dataId) as TypedArray;
  const weightsVals = backend.readSync(weights.dataId) as TypedArray;

  const outVals =
      bincountImplCPU(xVals, weightsVals, weights.dtype, weights.shape, size);

  return backend.makeTensorInfo([size], weights.dtype, outVals);
}

export const bincountConfig: KernelConfig = {
  kernelName: Bincount,
  backendName: 'webgl',
  kernelFunc: bincount as {} as KernelFunc
};
