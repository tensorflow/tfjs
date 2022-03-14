/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import {backend_util, Cumprod, CumprodAttrs, CumprodInputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {CumProdProgram} from '../cumprod_gpu';

import {identity} from './Identity';
import {transpose} from './Transpose';

export function cumprod(
    args: {inputs: CumprodInputs, backend: MathBackendWebGL,
           attrs: CumprodAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {axis, exclusive, reverse} = attrs;

  const xRank = x.shape.length;
  const permutation = backend_util.getAxesPermutation([axis], xRank);
  let permutedX = x;
  if (permutation != null) {
    permutedX = transpose({inputs: {x}, backend, attrs: {perm: permutation}});
  }
  const permutedAxis = backend_util.getInnerMostAxes(1, xRank)[0];

  if (permutedAxis !== xRank - 1) {
    throw new Error(
        `WebGL cumprod shader expects an inner-most axis=${
            x.shape.length - 1} ` +
        `but got axis=${axis}`);
  }
  const size = permutedX.shape[permutedAxis];
  let result = identity({inputs: {x: permutedX}, backend});
  // Use cumprod parallel algorithm, inspired by:
  // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
  // Note: although the algorithm is called sum, it works for any associtative
  // operator with an identity.

  for (let i = 0; i <= Math.ceil(Math.log2(size)) - 1; i++) {
    const program = new CumProdProgram(permutedX.shape, false, reverse);
    const customValues = [[i]];
    const prevResult = result;
    result =
        backend.runWebGLProgram(program, [result], result.dtype, customValues);
    backend.disposeIntermediateTensorInfo(prevResult);
  }
  // For exclusive cumprod, shift the end result in the direction of product
  // and add 1 to the front index.
  if (exclusive) {
    const program = new CumProdProgram(permutedX.shape, exclusive, reverse);
    const prevResult = result;
    result = backend.runWebGLProgram(program, [result], result.dtype);
    backend.disposeIntermediateTensorInfo(prevResult);
  }

  if (permutation != null) {
    const reversePermutation = backend_util.getUndoAxesPermutation(permutation);
    const reverseTransposedResult = transpose(
        {inputs: {x: result}, backend, attrs: {perm: reversePermutation}});

    backend.disposeIntermediateTensorInfo(result);
    backend.disposeIntermediateTensorInfo(permutedX);

    return reverseTransposedResult;
  }

  return result;
}

export const cumprodConfig: KernelConfig = {
  kernelName: Cumprod,
  backendName: 'webgl',
  kernelFunc: cumprod as {} as KernelFunc
};
