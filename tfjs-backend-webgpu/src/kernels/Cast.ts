/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import {BinaryInputs, Cast, CastAttrs, CastInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';

import {identity} from './Identity';
import {notEqual} from './NotEqual';

import {int} from '../kernel_utils/int';

export function cast(
    args: {inputs: CastInputs, backend: WebGPUBackend, attrs: CastAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {dtype} = attrs;

  if (!util.hasEncodingLoss(x.dtype, dtype)) {
    // We don't change the underlying data, since we cast to higher
    // precision.
    const result = identity({inputs: {x}, backend});
    return {dataId: result.dataId, shape: result.shape, dtype};
  }

  if (dtype === 'int32') {
    return int(x, backend);
  }

  if (dtype === 'bool') {
    const zerosTensorInfo = backend.makeTensorInfo(
        [], 'bool', util.getTypedArrayFromDType('bool', 1));

    const binaryInputs: BinaryInputs = {a: x, b: zerosTensorInfo};

    const result = notEqual({inputs: binaryInputs, backend}) as TensorInfo;
    return result;
  }

  throw new Error(`Error in Cast: failed to cast ${x.dtype} to ${dtype}`);
}

export const castConfig: KernelConfig = {
  kernelName: Cast,
  backendName: 'webgpu',
  kernelFunc: cast as {} as KernelFunc
};
