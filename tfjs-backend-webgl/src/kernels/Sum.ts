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

import {KernelConfig, KernelFunc, Sum, SumAttrs, SumInputs} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';

import {sumImpl} from './Sum_impl';

export function sum(
    args: {inputs: SumInputs, attrs: SumAttrs, backend: MathBackendWebGL}) {
  const {inputs, backend, attrs} = args;

  const {x} = inputs;
  const {axis, keepDims} = attrs;

  return sumImpl(x, axis, keepDims, backend);
}

export const sumConfig: KernelConfig = {
  kernelName: Sum,
  backendName: 'webgl',
  kernelFunc: sum as {} as KernelFunc
};
