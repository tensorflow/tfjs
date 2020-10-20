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

import {Fill, FillAttrs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {fillValues} from '../utils/fill_utils';

export function fill(args: {backend: MathBackendCPU, attrs: FillAttrs}):
    TensorInfo {
  const {backend, attrs} = args;
  const {shape, value, dtype} = attrs;

  const $dtype = dtype || util.inferDtype(value);
  const values = util.getArrayFromDType($dtype, util.sizeFromShape(shape));
  fillValues(values, value, $dtype);

  return backend.makeTensorInfo(shape, $dtype, values);
}

export const fillConfig: KernelConfig = {
  kernelName: Fill,
  backendName: 'cpu',
  kernelFunc: fill as {} as KernelFunc
};
