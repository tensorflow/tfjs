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

import {ClipByValue, ClipByValueAttrs, ClipByValueInputs, env, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {ClipProgram} from '../clip_gpu';
import {ClipPackedProgram} from '../clip_packed_gpu';

export function clipByValue(args: {
  inputs: ClipByValueInputs,
  backend: MathBackendWebGL,
  attrs: ClipByValueAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {clipValueMin, clipValueMax} = attrs;

  let program;
  if (env().getBool('WEBGL_PACK_CLIP')) {
    program = new ClipPackedProgram(x.shape);
  } else {
    program = new ClipProgram(x.shape);
  }
  const customSetup = program.getCustomSetupFunc(clipValueMin, clipValueMax);
  return backend.runWebGLProgram(program, [x], x.dtype, customSetup);
}

export const clipByValueConfig: KernelConfig = {
  kernelName: ClipByValue,
  backendName: 'webgl',
  kernelFunc: clipByValue as {} as KernelFunc
};
