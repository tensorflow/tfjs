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

import {env, KernelConfig, KernelFunc, NumericDataType, TensorInfo, TopK, TopKAttrs, TopKInputs, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {topKImplCPU} from '../kernel_utils/shared';
import {MergeProgram, SwapProgram} from '../top_k_gpu';
import {fill} from './Fill';
import {gatherV2} from './GatherV2';
import {reshape} from './Reshape';
import {slice} from './Slice';

function disposeIntermediateTensorInfoOrNull(
    backend: MathBackendWebGL, tensorInfo: TensorInfo) {
  if (tensorInfo !== null) {
    backend.disposeIntermediateTensorInfo(tensorInfo);
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
    args: {inputs: TopKInputs, backend: MathBackendWebGL, attrs: TopKAttrs}):
    TensorInfo[] {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {k, sorted} = attrs;

  // Empirically determined constant used to determine last dim threshold for
  // handing off execution to the CPU.
  const TOPK_LAST_DIM_CPU_HANDOFF_SIZE_THRESHOLD =
      env().getNumber('TOPK_LAST_DIM_CPU_HANDOFF_SIZE_THRESHOLD');

  // Empirically determined constant used to determine k threshold for handing
  // off execution to the CPU.
  const TOPK_K_CPU_HANDOFF_THRESHOLD =
      env().getNumber('TOPK_K_CPU_HANDOFF_THRESHOLD');

  const xShape = x.shape;
  const lastDim = xShape[xShape.length - 1];

  if (backend.shouldExecuteOnCPU([x]) ||
      lastDim < TOPK_LAST_DIM_CPU_HANDOFF_SIZE_THRESHOLD ||
      k > TOPK_K_CPU_HANDOFF_THRESHOLD) {
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

  // Eagerly unpack x input since it is passed in to all the shaders which
  // require unpacked inputs.
  const xtexData = backend.texData.get(x.dataId);
  const xIsPacked = xtexData !== null && xtexData.isPacked;
  const xUnPacked = xIsPacked ? backend.unpackTensor(x) : x;

  // Reshape into a 2d tensor [batch, lastDim] and compute topk along lastDim.
  const xSize = util.sizeFromShape(xShape);
  const batch = xSize / lastDim;
  const x2D = reshape(
      {inputs: {x: xUnPacked}, attrs: {shape: [batch, lastDim]}, backend});

  if (xIsPacked) {
    disposeIntermediateTensorInfoOrNull(backend, xUnPacked);
  }

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
    const fistPass = indices === null ? 1 : 0;
    const customValues =
        [[lastDim], [fistPass], [Number.NEGATIVE_INFINITY], [dir], [inc]];
    const prevIndices = indices;
    indices = backend.runWebGLProgram(program, inputs, 'int32', customValues);
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
    const customValues = [[lastDim], [firstPass], [kPow2]];
    const prevIndices = indices;
    indices =
        backend.runWebGLProgram(mergeProgram, inputs, 'int32', customValues);
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
  backendName: 'webgl',
  kernelFunc: topK as {} as KernelFunc
};
