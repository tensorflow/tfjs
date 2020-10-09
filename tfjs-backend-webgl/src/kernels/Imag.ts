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

import {Imag, ImagInputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {identity} from './Identity';

export function imag(args: {inputs: ImagInputs, backend: MathBackendWebGL}):
    TensorInfo {
  const {inputs, backend} = args;
  const {input} = inputs;
  const inputData = backend.texData.get(input.dataId);

  return identity({inputs: {x: inputData.complexTensorInfos.imag}, backend});
}

export const imagConfig: KernelConfig = {
  kernelName: Imag,
  backendName: 'webgl',
  kernelFunc: imag as {} as KernelFunc
};
