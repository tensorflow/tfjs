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

import {KernelConfig, TensorInfo, Tensor, Transpose, TransposeAttrs, TransposeInputs, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';

import {transposeImpl} from './Transpose_impl';
import {transposeSharedImpl} from './Transpose_shared_impl';

export const transposeConfig: KernelConfig = {
  kernelName: Transpose,
  backendName: 'webgpu',
  kernelFunc: ({inputs, attrs, backend}) => {
    const {x} = inputs as TransposeInputs;
    const {perm} = attrs as {} as TransposeAttrs;
    const webgpuBackend = backend as WebGPUBackend;

    const xRank = x.shape.length;

    const newShape: number[] = new Array(xRank);
    for (let i = 0; i < newShape.length; i++) {
      newShape[i] = x.shape[perm[i]];
    }

    let out: TensorInfo;
    if (webgpuBackend.shouldExecuteOnCPU([x as Tensor])) {
      out = webgpuBackend.cpuBackend.transpose(x as Tensor, perm);
      return out;
    }
    if (x.shape.length === 2 && util.arraysEqual(perm, [1, 0])) {
      out = transposeSharedImpl(x, perm, webgpuBackend);
      return out;
    }
    out = transposeImpl(x, perm, webgpuBackend);
    return out;
  }
};
