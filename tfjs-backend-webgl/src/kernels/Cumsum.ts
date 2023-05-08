/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import {Cumsum, CumsumAttrs, CumsumInputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {CumOpType} from '../cum_gpu';
import {cumImpl} from './Cum_impl';

export function cumsum(
    args:
        {inputs: CumsumInputs, backend: MathBackendWebGL, attrs: CumsumAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {axis, exclusive, reverse} = attrs;
  return cumImpl(CumOpType.Sum, x, backend, axis, exclusive, reverse);
}

export const cumsumConfig: KernelConfig = {
  kernelName: Cumsum,
  backendName: 'webgl',
  kernelFunc: cumsum as unknown as KernelFunc
};
