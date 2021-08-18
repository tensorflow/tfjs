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

import {KernelConfig, KernelFunc, OnesLike, OnesLikeInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {complex} from './Complex';
import {fill} from './Fill';
import {imag} from './Imag';
import {real} from './Real';
import {zerosLike} from './ZerosLike';

export function onesLike(
    args: {inputs: OnesLikeInputs, backend: MathBackendCPU}): TensorInfo {
  const {inputs, backend} = args;
  const {x} = inputs;

  if (x.dtype === 'string') {
    throw new Error('onesLike is not supported for string tensors');
  } else if (x.dtype === 'complex64') {
    const realPart = real({inputs: {input: x}, backend});
    const r = onesLike({inputs: {x: realPart}, backend});
    const imagPart = imag({inputs: {input: x}, backend});
    const i = zerosLike({inputs: {x: imagPart}, backend});

    const result = complex({inputs: {real: r, imag: i}, backend});

    backend.disposeIntermediateTensorInfo(realPart);
    backend.disposeIntermediateTensorInfo(r);
    backend.disposeIntermediateTensorInfo(imagPart);
    backend.disposeIntermediateTensorInfo(i);

    return result;
  } else {
    return fill({backend, attrs: {shape: x.shape, value: 1, dtype: x.dtype}});
  }
}

export const onesLikeConfig: KernelConfig = {
  kernelName: OnesLike,
  backendName: 'cpu',
  kernelFunc: onesLike as {} as KernelFunc
};
