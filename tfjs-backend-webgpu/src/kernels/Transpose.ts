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

import {engine, KernelConfig, KernelFunc, Tensor, Transpose, TransposeAttrs, TransposeInputs, TypedArray, util} from '@tensorflow/tfjs-core';
import {WebGPUBackend} from '../backend_webgpu';
import {transposeImplCPU as cpuTranspose} from '../kernel_utils/shared';
import {TransposeProgram} from './transpose_webgpu';
import {TransposeSharedProgram} from './transpose_shared_webgpu';

export function transpose(args: {
  inputs: TransposeInputs,
  attrs: TransposeAttrs,
  backend: WebGPUBackend 
}) {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {perm} = attrs;
  const webgpuBackend = backend;

  const xRank = x.shape.length;
  const newShape: number[] = new Array(xRank);
  for (let i = 0; i < newShape.length; i++) {
    newShape[i] = x.shape[perm[i]];
  }
  if (backend.shouldExecuteOnCPU([x as Tensor])) {
    const xTexData = webgpuBackend.tensorMap.get(x.dataId);
    const values = xTexData.values as TypedArray;
    const outValues = cpuTranspose(values, x.shape, x.dtype, perm, newShape);
      const dataId = webgpuBackend.write(outValues, newShape, x.dtype);
      const output =
          engine().makeTensorFromDataId(dataId, newShape, x.dtype, webgpuBackend);
    return output;
  }
  if (x.shape.length === 2 && util.arraysEqual(perm, [1, 0])) {
    const program = new TransposeSharedProgram(x.shape, perm);
    return webgpuBackend.compileAndRun(program, [x]);
  }
  const program = new TransposeProgram(x.shape, perm);
  return webgpuBackend.compileAndRun(program, [x]);
}

export const transposeConfig: KernelConfig = {
  kernelName: Transpose,
  backendName: 'webgpu',
  kernelFunc: transpose as {} as KernelFunc
};
