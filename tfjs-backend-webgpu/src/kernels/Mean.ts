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

import {backend_util, KernelConfig, KernelFunc, Mean, MeanAttrs, MeanInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {reduce} from '../kernel_utils/reduce';
import {reshape} from '../kernels/Reshape';
import {transpose} from './Transpose';

export function mean(
    args: {inputs: MeanInputs, attrs: MeanAttrs, backend: WebGPUBackend}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {keepDims, axis} = attrs;
  const xRank = x.shape.length;
  const toDispose = [];

  const origAxes = util.parseAxisParam(axis, x.shape);
  let axes = origAxes;
  const permutedAxes = backend_util.getAxesPermutation(axes, xRank);

  let meanInput = x;
  if (permutedAxes != null) {
    meanInput = transpose({inputs: {x}, attrs: {perm: permutedAxes}, backend});
    axes = backend_util.getInnerMostAxes(axes.length, xRank);
    toDispose.push(meanInput);
  }

  backend_util.assertAxesAreInnerMostDims('mean', axes, xRank);

  const [meanOutShape, reduceShape] =
      backend_util.computeOutAndReduceShapes(meanInput.shape, axes);
  let outShape = meanOutShape;
  if (keepDims) {
    // rather than reshape at the end, set the target shape here.
    outShape = backend_util.expandShapeToKeepDim(meanOutShape, origAxes);
  }

  const inSize = util.sizeFromShape(reduceShape);
  const xSize = util.sizeFromShape(meanInput.shape);
  const batchSize = xSize / inSize;
  const reshapedInput = reshape(
      {inputs: {x: meanInput}, attrs: {shape: [batchSize, inSize]}, backend});
  toDispose.push(reshapedInput);
  const reduced = reduce(reshapedInput, 'float32', 'mean', backend);
  toDispose.push(reduced);
  const out = reshape({inputs: {x: reduced}, attrs: {shape: outShape},
      backend});

  toDispose.forEach(t => backend.disposeData(t.dataId));

  return out;
}

export const meanConfig: KernelConfig = {
  kernelName: Mean,
  backendName: 'webgpu',
  kernelFunc: mean as {} as KernelFunc
};
