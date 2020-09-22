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

import {KernelConfig, KernelFunc, Real, RealInputs, TensorInfo, TypedArray} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';

export function real(args: {inputs: RealInputs, backend: MathBackendWebGL}):
    TensorInfo {
  const {inputs, backend} = args;
  const {input} = inputs;

  // TODO(annxingyuan): Share data buckets once soft disposal through engine is
  // possible
  const vals = backend.readSync(input.dataId) as TypedArray;
  const dataId = backend.write(vals, input.shape, 'float32');
  const tensorInfo: TensorInfo = {dataId, shape: input.shape, dtype: 'float32'};

  // When complex tensor is disposed, its underlying parts will be disposed too.
  // Make new tensor out of the real values of the complex tensor. This makes
  // sure the value is still accessible even if complex tensor is disposed.
  return tensorInfo;
}

export const realConfig: KernelConfig = {
  kernelName: Real,
  backendName: 'webgl',
  kernelFunc: real as {} as KernelFunc
};
