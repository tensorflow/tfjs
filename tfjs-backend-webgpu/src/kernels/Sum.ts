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

import {KernelFunc, Sum, SumAttrs, SumInputs, sumOutType, TensorInfo} from '@tensorflow/tfjs-core';
import {backend_util, KernelConfig} from '@tensorflow/tfjs-core';
import {util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {reduce} from '../kernel_utils/reduce';
import {reshape} from './Reshape';

import {transpose} from './Transpose';
export function sum(
    args: {inputs: SumInputs, backend: WebGPUBackend, attrs: SumAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {axis, keepDims} = attrs;
  const webgpuBackend = backend;
  const xShape = x.shape;
  const xRank = xShape.length;

  const origAxes = util.parseAxisParam(axis, xShape);
  let axes = origAxes;
  const permutedAxes = backend_util.getAxesPermutation(axes, xRank);
  const sumInputIsTransposed = permutedAxes != null;

  let sumInput = x;
  if (sumInputIsTransposed) {
    sumInput = transpose({inputs: {x}, backend, attrs: {perm: permutedAxes}});

    axes = backend_util.getInnerMostAxes(axes.length, xRank);
  }
  backend_util.assertAxesAreInnerMostDims('sum', axes, xRank);
  const [sumOutShape, reduceShape] =
      backend_util.computeOutAndReduceShapes(sumInput.shape, axes);

  let outShape = sumOutShape;
  if (keepDims) {
    outShape = backend_util.expandShapeToKeepDim(sumOutShape, origAxes);
  }
  const reduceSize = util.sizeFromShape(reduceShape);
  const xSize = util.sizeFromShape(xShape);
  const batchSize = xSize / reduceSize;
  const a2D = reshape({
    inputs: {x: sumInput},
    attrs: {shape: [batchSize, reduceSize]},
    backend: webgpuBackend
  });
  const outputDType = sumOutType(x.dtype);
  const a2DReduce = reduce(a2D, outputDType, 'sum', webgpuBackend);
  const out = reshape({
    inputs: {x: a2DReduce},
    attrs: {shape: outShape},
    backend: webgpuBackend
  });

  backend.disposeData(a2D.dataId);
  backend.disposeData(a2DReduce.dataId);
  if (sumInputIsTransposed) {
    backend.disposeData(sumInput.dataId);
  }

  return out;
}

export const sumConfig: KernelConfig = {
  kernelName: Sum,
  backendName: 'webgpu',
  kernelFunc: sum as {} as KernelFunc
};
