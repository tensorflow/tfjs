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

import {env, KernelConfig, KernelFunc, MirrorPad, MirrorPadAttrs, MirrorPadInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {MirrorPadProgram} from '../mirror_pad_gpu';
import {MirrorPadPackedProgram} from '../mirror_pad_packed_gpu';

export const mirrorPadKernelFunc: (params: {
  inputs: MirrorPadInputs,
  backend: MathBackendWebGL,
  attrs: MirrorPadAttrs
}) => TensorInfo = ({inputs, backend, attrs}) => {
  const {x} = inputs;
  const {paddings, mode} = attrs;

  const program = env().getBool('WEBGL_PACK_ARRAY_OPERATIONS') ?
      new MirrorPadPackedProgram(x.shape, paddings, mode) :
      new MirrorPadProgram(x.shape, paddings, mode);

  const output = backend.runWebGLProgram(program, [x], x.dtype);

  return output;
};

export const mirrorPadConfig: KernelConfig = {
  kernelName: MirrorPad,
  backendName: 'webgl',
  kernelFunc: mirrorPadKernelFunc as {} as KernelFunc,
};
