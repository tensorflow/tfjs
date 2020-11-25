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

import {Diag, DiagInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {DiagProgram} from '../diag_gpu';
import {reshape} from './Reshape';

export function diag(args: {inputs: DiagInputs, backend: MathBackendWebGL}):
    TensorInfo {
  const {inputs, backend} = args;
  const {x} = inputs;

  const outShape = [...x.shape, ...x.shape];
  const xSize = util.sizeFromShape(x.shape);

  const flat = reshape({inputs: {x}, backend, attrs: {shape: [xSize]}});

  const program = new DiagProgram(xSize);
  const res = backend.runWebGLProgram(program, [flat], flat.dtype);

  const out = reshape({inputs: {x: res}, backend, attrs: {shape: outShape}});

  backend.disposeIntermediateTensorInfo(flat);
  backend.disposeIntermediateTensorInfo(res);

  return out;
}

export const diagConfig: KernelConfig = {
  kernelName: Diag,
  backendName: 'webgl',
  kernelFunc: diag as {} as KernelFunc
};
