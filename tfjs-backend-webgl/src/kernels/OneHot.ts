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

import {KernelConfig, KernelFunc, OneHot, OneHotAttrs, OneHotInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {OneHotProgram} from '../onehot_gpu';
import {reshape} from './Reshape';

export const oneHot = (args: {
  inputs: OneHotInputs,
  backend: MathBackendWebGL,
  attrs: OneHotAttrs
}): TensorInfo => {
  const {inputs, backend, attrs} = args;
  const {indices} = inputs;
  const {depth, onValue, offValue} = attrs;

  const indicesSize = util.sizeFromShape(indices.shape);
  const program = new OneHotProgram(indicesSize, depth, onValue, offValue);
  const reshaped =
      reshape({inputs: {x: indices}, backend, attrs: {shape: [indicesSize]}});
  const result = backend.runWebGLProgram(program, [reshaped], indices.dtype);
  backend.disposeIntermediateTensorInfo(reshaped);

  const outShape = [...indices.shape, depth];
  const out = reshape({inputs: {x: result}, backend, attrs: {shape: outShape}});
  backend.disposeIntermediateTensorInfo(result);
  return out;
};

export const oneHotConfig: KernelConfig = {
  kernelName: OneHot,
  backendName: 'webgl',
  kernelFunc: oneHot as {} as KernelFunc
};
