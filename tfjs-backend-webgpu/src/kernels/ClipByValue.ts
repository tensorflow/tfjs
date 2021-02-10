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

import {ClipByValue, ClipByValueAttrs, ClipByValueInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';

import {ClipVec4Program} from './clip_vec4_webgpu';
import {ClipProgram} from './clip_webgpu';

export function clipByValue(args: {
  inputs: ClipByValueInputs,
  backend: WebGPUBackend,
  attrs: ClipByValueAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {clipValueMin, clipValueMax} = attrs;

  let program: ClipProgram|ClipVec4Program;
  if (util.sizeFromShape(x.shape) % 4 === 0) {
    program = new ClipVec4Program(x.shape, clipValueMin, clipValueMax);
  } else {
    program = new ClipProgram(x.shape, clipValueMin, clipValueMax);
  }
  return backend.runWebGPUProgram(program, [x], x.dtype);
}

export const clipByValueConfig: KernelConfig = {
  kernelName: ClipByValue,
  backendName: 'webgpu',
  kernelFunc: clipByValue as {} as KernelFunc
};
