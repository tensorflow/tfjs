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

import {Atan2, Atan2Inputs, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';
import {KernelConfig} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';

import {atan2Impl} from './Atan2_impl';

export const atan2KernelFunc:
    (params: {inputs: Atan2Inputs, backend: MathBackendWebGL}) =>
        TensorInfo | TensorInfo[] = ({inputs, backend}) => {
          const {a, b} = inputs;
          return atan2Impl(a, b, backend);
        };

export const atan2Config: KernelConfig = {
  kernelName: Atan2,
  backendName: 'webgl',
  kernelFunc: atan2KernelFunc as {} as KernelFunc,
};
