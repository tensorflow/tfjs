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
import {backend_util, KernelConfig, KernelFunc, MaxPool, MaxPoolAttrs, MaxPoolInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {Pool2DProgram} from '../pool_gpu';
import {assertNotComplex} from '../webgl_util';
import {identity} from './Identity';

export function maxPool(args: {
  inputs: MaxPoolInputs,
  backend: MathBackendWebGL,
  attrs: MaxPoolAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  assertNotComplex(x, 'maxPool');
  const {filterSize, strides, pad, dimRoundingMode} = attrs;
  const dilations = 1;

  util.assert(
      backend_util.eitherStridesOrDilationsAreOne(strides, dilations),
      () => 'Error in maxPool: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);

  const convInfo = backend_util.computePool2DInfo(
      x.shape as [number, number, number, number], filterSize, strides,
      dilations, pad, dimRoundingMode);
  if (convInfo.filterWidth === 1 && convInfo.filterHeight === 1 &&
      util.arraysEqual(convInfo.inShape, convInfo.outShape)) {
    return identity({inputs: {x}, backend});
  }
  const maxPoolProgram = new Pool2DProgram(convInfo, 'max', false);
  return backend.runWebGLProgram(maxPoolProgram, [x], x.dtype);
}

export const maxPoolConfig: KernelConfig = {
  kernelName: MaxPool,
  backendName: 'webgl',
  kernelFunc: maxPool as {} as KernelFunc
};
