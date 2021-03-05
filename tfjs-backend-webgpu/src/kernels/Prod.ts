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

import {backend_util, KernelConfig, KernelFunc, Prod, ProdAttrs, ProdInputs, sumOutType, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {prodImplCPU} from '../kernel_utils/shared';
import {reduce} from '../kernel_utils/reduce';
import {reshape} from './Reshape';
import {transpose} from './Transpose';

export function prod(
    args: {inputs: ProdInputs, backend: WebGPUBackend, attrs: ProdAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {axis, keepDims} = attrs;
  const xRank = x.shape.length;
  const toDispose = [];

  const origAxes = util.parseAxisParam(axis, x.shape);
  let axes = origAxes;
  const permutedAxes = backend_util.getAxesPermutation(axes, xRank);

  let prodInput = x;
  if (permutedAxes != null) {
    prodInput = transpose({inputs: {x}, backend, attrs: {perm: permutedAxes}});
    axes = backend_util.getInnerMostAxes(axes.length, xRank);
    toDispose.push(prodInput);
  }

  backend_util.assertAxesAreInnerMostDims('prod', axes, xRank);

  let res;
  if (backend.shouldExecuteOnCPU([prodInput])) {
    const xVals = backend.tensorMap.get(prodInput.dataId).values as TypedArray;
    const {outVals, outShape, outDtype} =
        prodImplCPU(prodInput.shape, prodInput.dtype, xVals, axes);
    res = backend.makeTensorInfo(outShape, outDtype, outVals);
  } else {
    const [prodOutShape, reduceShape] =
        backend_util.computeOutAndReduceShapes(prodInput.shape, axes);
    let outShape = prodOutShape;
    if (keepDims) {
      // rather than reshape at the end, set the target shape here.
      outShape = backend_util.expandShapeToKeepDim(prodOutShape, origAxes);
    }

    const inSize = util.sizeFromShape(reduceShape);
    const xSize = util.sizeFromShape(x.shape);
    const batchSize = xSize / inSize;
    const reshapedInput = reshape(
        {inputs: {x: prodInput}, backend, attrs: {shape: [batchSize, inSize]}});
    toDispose.push(reshapedInput);
    const outputDType = sumOutType(x.dtype);
    const reduced = reduce(reshapedInput, outputDType, 'prod', backend);
    toDispose.push(reduced);
    res = reshape({inputs: {x: reduced}, backend, attrs: {shape: outShape}});
  }

  toDispose.forEach(t => backend.disposeData(t.dataId));

  return res;
}

export const prodConfig: KernelConfig = {
  kernelName: Prod,
  backendName: 'webgpu',
  kernelFunc: prod as {} as KernelFunc
};
