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

import {env, KernelConfig, KernelFunc, Reverse, ReverseAttrs, ReverseInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {ReverseProgram} from '../reverse_gpu';
import {ReversePackedProgram} from '../reverse_packed_gpu';

import {identity} from './Identity';

export function reverse(args: {
  inputs: ReverseInputs,
  backend: MathBackendWebGL,
  attrs: ReverseAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {dims} = attrs;

  const xRank = x.shape.length;

  const $dims = util.parseAxisParam(dims, x.shape);
  if (xRank === 0) {
    return identity({inputs: {x}, backend});
  }

  const program = env().getBool('WEBGL_PACK_ARRAY_OPERATIONS') ?
      new ReversePackedProgram(x.shape, $dims) :
      new ReverseProgram(x.shape, $dims);

  return backend.runWebGLProgram(program, [x], x.dtype);
}

export const reverseConfig: KernelConfig = {
  kernelName: Reverse,
  backendName: 'webgl',
  kernelFunc: reverse as {} as KernelFunc
};
