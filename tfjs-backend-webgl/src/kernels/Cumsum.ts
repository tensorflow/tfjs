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

import {backend_util, Cumsum, CumsumAttrs, CumsumInputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {CumSumProgram} from '../cumsum_gpu';

import {identity} from './Identity';
import {transpose} from './Transpose';

export function cumsum(
    args:
        {inputs: CumsumInputs, backend: MathBackendWebGL, attrs: CumsumAttrs}):
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
        `WebGL cumsum shader expects an inner-most axis=${
            x.shape.length - 1} ` +
        `but got axis=${axis}`);
  }
  const size = x.shape[permutedAxis];
  let result = identity({inputs: {x: permutedX}, backend});
  // Use cumsum parallel algorithm, ref:
  // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

  for (let i = 0; i <= Math.ceil(Math.log2(size)) - 1; i++) {
    const program = new CumSumProgram(permutedX.shape, false, reverse);
    const customSetup = program.getCustomSetupFunc(i);
    const prevResult = result;
    result =
        backend.runWebGLProgram(program, [result], result.dtype, customSetup);
    backend.disposeIntermediateTensorInfo(prevResult);
  }
  // For exclusive cumsum, shift the end result in the direction of sum
  // and add 0 to the front index.
  if (exclusive) {
    const program = new CumSumProgram(permutedX.shape, exclusive, reverse);
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

export const cumsumConfig: KernelConfig = {
  kernelName: Cumsum,
  backendName: 'webgl',
  kernelFunc: cumsum as {} as KernelFunc
};
