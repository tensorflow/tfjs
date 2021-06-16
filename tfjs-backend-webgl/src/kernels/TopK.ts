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

import {KernelConfig, KernelFunc, TensorInfo, TopK, TopKAttrs, TopKInputs, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {MergeProgram, SwapProgram} from '../top_k_gpu';
import { slice } from './Slice';

export function topK(
    args: {inputs: TopKInputs, backend: MathBackendWebGL, attrs: TopKAttrs}):
    TensorInfo[] {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const k = attrs.k;

  const xShape = x.shape;
  // Reshape into a 2d tensor [batch, lastDim] and compute topk along lastDim.
  const lastDim = xShape[xShape.length - 1];
  const xSize = util.sizeFromShape(xShape);
  const batch = xSize / lastDim;
  x.shape = [batch, lastDim];
  let result = x;

  // Step 1: local sort
  for (let len = 1; len < k; len *= 2) {
    const dir = len * 2;
    for (let inc = len; inc > 0; inc /= 2) {
      const program = new SwapProgram([2, batch, lastDim], result.shape.length === 2 ? 2 : 3);
      const customSetup = program.getCustomSetupFunc(dir, inc);
      const prevResult = result;
      result =
          backend.runWebGLProgram(program, [result], result.dtype, customSetup);
      backend.disposeIntermediateTensorInfo(prevResult);
    }
  }

  for (let resultSize = lastDim; resultSize > k; resultSize /= 2) {
    // Step 2: merge
    const mergeProgram = new MergeProgram([2, batch, resultSize / 2], result.shape.length === 2 ? 2 : 3);
    let customSetup = mergeProgram.getCustomSetupFunc(k);
    let prevResult = result;
    result = backend.runWebGLProgram(
        mergeProgram, [result], result.dtype, customSetup);
    backend.disposeIntermediateTensorInfo(prevResult);

    const len = Math.floor(k / 2);
    const dir = len * 2;
    for (let inc = len; inc > 0; inc /= 2) {
      // Step 3: rebuild
      const swapProgram = new SwapProgram(result.shape, result.shape.length === 2 ? 2 : 3);
      customSetup = swapProgram.getCustomSetupFunc(dir, inc);
      prevResult = result;
      result = backend.runWebGLProgram(
          swapProgram, [result], result.dtype, customSetup);
      backend.disposeIntermediateTensorInfo(prevResult);
    }
  }

  const values = slice({inputs: {x: result}, backend, attrs: {begin: [0, 0, 0], size: [1, batch, k]}});
  const indices = slice({inputs: {x: result}, backend, attrs: {begin: [1, 0, 0], size: [1, batch, k]}});
  backend.disposeIntermediateTensorInfo(result);

  // Reshape back to the original input shape, except that the last
  // dimension is k.
  xShape[xShape.length - 1] = k;
  values.shape = xShape;
  indices.shape = xShape;

  indices.dtype = 'int32';

  return [values, indices];
}

export const topKConfig: KernelConfig = {
  kernelName: TopK,
  backendName: 'webgl',
  kernelFunc: topK as {} as KernelFunc
};
