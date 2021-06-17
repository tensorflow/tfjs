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
import {fill} from './Fill';
import {gatherV2} from './GatherV2';
import {slice} from './Slice';

function roundUpToPow2(num: number) {
  let pow2 = 1;
  while (pow2 < num) {
    pow2 *= 2;
  }
  return pow2;
}

// Based on Algorithm 2 of Bitonic Top K, ref:
// https://anilshanbhag.in/static/papers/gputopk_sigmod18.pdf
export function topK(
    args: {inputs: TopKInputs, backend: MathBackendWebGL, attrs: TopKAttrs}):
    TensorInfo[] {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const k = attrs.k;

  const xShape = x.shape;
  const lastDim = xShape[xShape.length - 1];

  if (k === 0) {
    xShape[xShape.length - 1] = 0;
    return [
      backend.makeTensorInfo(xShape, x.dtype, []),
      backend.makeTensorInfo(xShape, 'int32', [])
    ];
  }

  if (lastDim === 1) {
    return [
      x, fill({attrs: {shape: xShape, dtype: 'int32', value: 0}, backend})
    ];
  }

  // Reshape into a 2d tensor [batch, lastDim] and compute topk along lastDim.
  const xSize = util.sizeFromShape(xShape);
  const batch = xSize / lastDim;
  x.shape = [batch, lastDim];

  const kPow2 = roundUpToPow2(k);
  const lastDimPow2 = roundUpToPow2(lastDim);

  let indices: TensorInfo = null;

  const getInputs = () => indices === null ? [x] : [x, indices];

  const disposeIntermediateTensorInfoOrNull = (tensorInfo: TensorInfo) => {
    if (tensorInfo !== null) {
      backend.disposeIntermediateTensorInfo(tensorInfo);
    }
  };

  const runSwap = (dir: number, inc: number, shape: number[]) => {
    const inputs = getInputs();
    const program = new SwapProgram(lastDim, shape, inputs.length === 1);
    const customSetup = program.getCustomSetupFunc(dir, inc);
    const prevIndices = indices;
    indices = backend.runWebGLProgram(program, inputs, 'int32', customSetup);
    disposeIntermediateTensorInfoOrNull(prevIndices);
  };

  // Step 1: local sort
  for (let len = 1; len < kPow2; len *= 2) {
    const dir = len * 2;
    for (let inc = len; inc >= 1; inc /= 2) {
      runSwap(dir, inc, [batch, lastDimPow2]);
    }
  }

  for (let indicesSize = lastDimPow2; indicesSize > kPow2; indicesSize /= 2) {
    // Step 2: merge
    const inputs = getInputs();
    const mergeProgram = new MergeProgram(
        lastDim, kPow2, [batch, indicesSize / 2], inputs.length === 1);
    const prevIndices = indices;
    indices = backend.runWebGLProgram(mergeProgram, inputs, 'int32');
    disposeIntermediateTensorInfoOrNull(prevIndices);

    const len = kPow2 / 2;
    const dir = len * 2;
    for (let inc = len; inc >= 1; inc /= 2) {
      // Step 3: rebuild
      runSwap(dir, inc, indices.shape);
    }
  }

  // Keep only the requested top K results instead of kPow2
  const prevIndices = indices;
  indices = slice(
      {inputs: {x: indices}, backend, attrs: {begin: 0, size: [batch, k]}});
  disposeIntermediateTensorInfoOrNull(prevIndices);

  const values =
      gatherV2({inputs: {x, indices}, backend, attrs: {axis: 1, batchDims: 1}});

  // Reshape back to the original input shape, except that the last
  // dimension is k.
  xShape[xShape.length - 1] = k;
  indices.shape = xShape;
  values.shape = xShape;

  return [values, indices];
}

export const topKConfig: KernelConfig = {
  kernelName: TopK,
  backendName: 'webgl',
  kernelFunc: topK as {} as KernelFunc
};
