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

import {Fill, FillAttrs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {FillProgram} from './fill_webgpu';

export function fill(args: {backend: WebGPUBackend, attrs: FillAttrs}):
    TensorInfo {
  const {backend, attrs} = args;
  const {shape, value} = attrs;
  let {dtype} = attrs;

  dtype = dtype || util.inferDtype(value);

  if (dtype === 'string') {
    // String type should be handled in CPU memory.
    const values = util.getArrayFromDType(dtype, util.sizeFromShape(shape));
    values.fill(value as string);
    return backend.makeTensorInfo(shape, dtype, values);
  } else {
    const program = new FillProgram(shape);
    const uniformData = new Float32Array([value as number]);
    return backend.runWebGPUProgram(program, [], dtype, uniformData);
  }
}

export const fillConfig: KernelConfig = {
  kernelName: Fill,
  backendName: 'webgpu',
  kernelFunc: fill as {} as KernelFunc
};
