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
import {AvgPool, AvgPoolAttrs, AvgPoolInputs, backend_util, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {Pool2DProgram} from '../pool_gpu';
import {assertNotComplex} from '../webgl_util';
import {identity} from './Identity';

export function avgPool(args: {
  inputs: AvgPoolInputs,
  backend: MathBackendWebGL,
  attrs: AvgPoolAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  assertNotComplex(x, 'avgPool');
  const {filterSize, strides, pad, dimRoundingMode} = attrs;
  const dilations = 1;

  util.assert(
      backend_util.eitherStridesOrDilationsAreOne(strides, dilations),
      () => 'Error in avgPool: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);

  const convInfo = backend_util.computePool2DInfo(
      x.shape as [number, number, number, number], filterSize, strides,
      dilations, pad, dimRoundingMode);
  if (convInfo.filterWidth === 1 && convInfo.filterHeight === 1 &&
      util.arraysEqual(convInfo.inShape, convInfo.outShape)) {
    return identity({inputs: {x}, backend});
  }
  const avgPoolProgram = new Pool2DProgram(convInfo, 'avg', false);
  return backend.runWebGLProgram(avgPoolProgram, [x], 'float32');
}

export const avgPoolConfig: KernelConfig = {
  kernelName: AvgPool,
  backendName: 'webgl',
  kernelFunc: avgPool as {} as KernelFunc
};
