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

import {AddN, AddNInputs, env, KernelConfig, KernelFunc, TensorInfo, upcastType} from '@tensorflow/tfjs-core';

import {AddNProgram} from '../addn_gpu';
import {AddNPackedProgram} from '../addn_packed_gpu';
import {MathBackendWebGL} from '../backend_webgl';
import {identity} from './Identity';

export function addN(args: {inputs: AddNInputs, backend: MathBackendWebGL}):
    TensorInfo {
  const {inputs, backend} = args;

  const tensors = inputs;
  if (tensors.length === 1) {
    return identity({inputs: {x: tensors[0]}, backend});
  }

  // Limit the number of uploaded textures for optimization.
  if (tensors.length > env().get('WEBGL_MAX_TEXTURES_IN_SHADER')) {
    const midIndex = Math.floor(tensors.length / 2);
    const leftSide = addN({inputs: tensors.slice(0, midIndex), backend});
    const rightSide = addN({inputs: tensors.slice(midIndex), backend});
    return addN({inputs: [leftSide, rightSide], backend});
  }

  const dtype =
      tensors.map(t => t.dtype).reduce((d1, d2) => upcastType(d1, d2));
  const shapes = tensors.map(t => t.shape);
  // We can make sure shapes are identical in op level.
  const usePackedOp = env().getBool('WEBGL_PACK');
  const program = usePackedOp ?
      new AddNPackedProgram(tensors[0].shape, shapes) :
      new AddNProgram(tensors[0].shape, shapes);
  return backend.runWebGLProgram(program, tensors, dtype);
}

export const addNConfig: KernelConfig = {
  kernelName: AddN,
  backendName: 'webgl',
  kernelFunc: addN as {} as KernelFunc
};
