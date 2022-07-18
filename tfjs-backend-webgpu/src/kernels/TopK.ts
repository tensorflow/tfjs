/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {KernelConfig, KernelFunc, NumericDataType, TensorInfo, TopK, TopKAttrs, TopKInputs, TypedArray, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {topKImplCPU} from '../kernel_utils/shared';
import {MergeProgram, SwapProgram} from '../top_k_webgpu';
import {fill} from './Fill';
import {gatherV2} from './GatherV2';
import {reshape} from './Reshape';
import {slice} from './Slice';

function disposeIntermediateTensorInfoOrNull(
    backend: WebGPUBackend, tensorInfo: TensorInfo) {
  if (tensorInfo !== null) {
    backend.disposeData(tensorInfo.dataId);
  }
}

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
    args: {inputs: TopKInputs, backend: WebGPUBackend, attrs: TopKAttrs}):
    TensorInfo[] {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {k, sorted}= attrs;

  const xShape = x.shape;
  const lastDim = xShape[xShape.length - 1];

  if (backend.shouldExecuteOnCPU([x])) {
    const xVals = backend.readSync(x.dataId) as TypedArray;
    const [allTopKVals, allTopKIndices] =
        topKImplCPU(xVals, xShape, x.dtype as NumericDataType, k, sorted);

    return [
      backend.makeTensorInfo(
          allTopKVals.shape, allTopKVals.dtype, allTopKVals.values),
      backend.makeTensorInfo(
          allTopKIndices.shape, allTopKIndices.dtype, allTopKIndices.values)
    ];
  }

  if (k === 0) {
    xShape[xShape.length - 1] = 0;
    return [
      backend.makeTensorInfo(xShape, x.dtype, []),
      backend.makeTensorInfo(xShape, 'int32', [])
    ];
  }

  if (lastDim === 1 /* firstPass */) {
    return [
      x, fill({attrs: {shape: xShape, dtype: 'int32', value: 0}, backend})
    ];
  }

  // Reshape into a 2d tensor [batch, lastDim] and compute topk along lastDim.
  const xSize = util.sizeFromShape(xShape);
  const batch = xSize / lastDim;
  const x2D = reshape({inputs: {x}, attrs: {shape: [batch, lastDim]}, backend});

  const kPow2 = roundUpToPow2(k);
  const lastDimPow2 = roundUpToPow2(lastDim);

  // Only the indices containing the top K are kept at every step to reduce
  // number of outputs in the GPU algorithms, so once the final set of indices
  // is computed then gather is used to grab the corresponding values
  // from the original input.
  let indices: TensorInfo = null;

  // GPU algorithm always takes in an indices input but this input is not used
  // on the first run of a GPU algorithm, therefore if indices is null we simply
  // pass in x2D instead of it but the value will not actually be used
  const getInputs = () => indices === null ? [x2D, x2D] : [x2D, indices];

  const runSwap = (dir: number, inc: number, shape: number[]) => {
    const inputs = getInputs();
    const program = new SwapProgram(shape);
    const firstPass = indices === null ? 1 : 0;
    const uniformDataSwap = [
        {type: 'int32', data: [lastDim]},
        {type: 'int32', data: [firstPass]},
        {type: 'float32', data: [Number.NEGATIVE_INFINITY]},
        {type: 'int32', data: [dir]},
        {type: 'int32', data: [inc]}
    ];
    const prevIndices = indices;
    indices = backend.runWebGPUProgram(
        program, inputs, 'int32', uniformDataSwap);
    disposeIntermediateTensorInfoOrNull(backend, prevIndices);
  };

  // Step 1: local sort
  for (let len = 1; len < kPow2; len *= 2) {
    const dir = len * 2;
    for (let inc = len; inc >= 1; inc /= 2) {
      runSwap(dir, inc, [batch, lastDimPow2]);
    }
  }

  // Step 2: merge
  for (let indicesSize = lastDimPow2; indicesSize > kPow2; indicesSize /= 2) {
    const inputs = getInputs();
    const mergeProgram = new MergeProgram([batch, indicesSize / 2]);
    const firstPass = indices === null ? 1 : 0;
    const uniformDataMerge = [
        {type: 'int32', data: [lastDim]},
        {type: 'int32', data: [firstPass]},
        {type: 'int32', data: [kPow2]}
    ];
    const prevIndices = indices;
    indices = backend.runWebGPUProgram(
        mergeProgram, inputs, 'int32', uniformDataMerge);
    disposeIntermediateTensorInfoOrNull(backend, prevIndices);

    // Step 3: rebuild
    const len = kPow2 / 2;
    const dir = len * 2;
    for (let inc = len; inc >= 1; inc /= 2) {
      runSwap(dir, inc, indices.shape);
    }
  }

  // Keep only the requested top K results instead of kPow2
  let prevIndices = indices;
  indices = slice(
      {inputs: {x: indices}, backend, attrs: {begin: 0, size: [batch, k]}});
  disposeIntermediateTensorInfoOrNull(backend, prevIndices);

  // Gather values on last dimension
  let values = gatherV2(
      {inputs: {x: x2D, indices}, backend, attrs: {axis: 1, batchDims: 1}});
  disposeIntermediateTensorInfoOrNull(backend, x2D);

  // Reshape back to the original input shape, except that the last
  // dimension is k.
  const newShape = xShape.slice(0, -1);
  newShape.push(k);

  prevIndices = indices;
  indices = reshape({inputs: {x: indices}, attrs: {shape: newShape}, backend});
  disposeIntermediateTensorInfoOrNull(backend, prevIndices);

  const prevValues = values;
  values = reshape({inputs: {x: values}, attrs: {shape: newShape}, backend});
  disposeIntermediateTensorInfoOrNull(backend, prevValues);

  return [values, indices];
}

export const topKConfig: KernelConfig = {
  kernelName: TopK,
  backendName: 'webgpu',
  kernelFunc: topK as {} as KernelFunc
};
