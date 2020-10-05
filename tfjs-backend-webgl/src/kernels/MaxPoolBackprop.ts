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
import {backend_util, KernelConfig, KernelFunc, MaxPoolBackprop, MaxPoolBackpropAttrs, MaxPoolBackpropInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {MaxPool2DBackpropProgram} from '../max_pool_backprop_gpu';
import {Pool2DProgram} from '../pool_gpu';
import {assertNotComplex} from '../webgl_util';

export function maxPoolBackprop(args: {
  inputs: MaxPoolBackpropInputs,
  backend: MathBackendWebGL,
  attrs: MaxPoolBackpropAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {dy, input, output} = inputs;
  const x = input;
  assertNotComplex([input, output], 'maxPoolBackprop');
  const {filterSize, strides, pad, dimRoundingMode} = attrs;

  const convInfo = backend_util.computePool2DInfo(
      x.shape as [number, number, number, number], filterSize, strides,
      1 /* dilations */, pad, dimRoundingMode);
  const getPositions = true;
  const maxPoolPositionsProgram =
      new Pool2DProgram(convInfo, 'max', getPositions);
  const maxPoolPositions: TensorInfo =
      backend.runWebGLProgram(maxPoolPositionsProgram, [x], x.dtype);

  const maxPoolBackPropProgram = new MaxPool2DBackpropProgram(convInfo);
  const result = backend.runWebGLProgram(
      maxPoolBackPropProgram, [dy, maxPoolPositions], x.dtype);
  backend.disposeIntermediateTensorInfo(maxPoolPositions);
  return result;
}

export const maxPoolBackpropConfig: KernelConfig = {
  kernelName: MaxPoolBackprop,
  backendName: 'webgl',
  kernelFunc: maxPoolBackprop as {} as KernelFunc
};
