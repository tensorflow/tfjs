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

import {backend_util, KernelConfig, KernelFunc, Prod, ProdAttrs, ProdInputs, sumOutType, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {reduce} from '../kernel_utils/reduce';
import {prodImplCPU} from '../kernel_utils/shared';

import {reshape} from './Reshape';
import {transpose} from './Transpose';

export function prod(
    args: {inputs: ProdInputs, backend: MathBackendWebGL, attrs: ProdAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {axis, keepDims} = attrs;

  const xRank = x.shape.length;
  const toDispose = [];

  const origAxes = util.parseAxisParam(axis, x.shape);
  let axes = origAxes;
  const permutedAxes = backend_util.getAxesPermutation(axes, xRank);
  let permutedX = x;
  if (permutedAxes != null) {
    permutedX = transpose({inputs: {x}, backend, attrs: {perm: permutedAxes}});
    axes = backend_util.getInnerMostAxes(axes.length, xRank);
    toDispose.push(permutedX);
  }

  backend_util.assertAxesAreInnerMostDims('prod', axes, xRank);

  let res;
  if (backend.shouldExecuteOnCPU([permutedX])) {
    const xVals = backend.texData.get(permutedX.dataId).values as TypedArray;
    const {outVals, outShape, outDtype} =
        prodImplCPU(permutedX.shape, permutedX.dtype, xVals, axes);
    res = backend.makeTensorInfo(outShape, outDtype, outVals);
  } else {
    const [outShape, reduceShape] =
        backend_util.computeOutAndReduceShapes(permutedX.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const a2D = reshape(
        {inputs: {x: permutedX}, backend, attrs: {shape: [-1, inSize]}});
    const outputDType = sumOutType(x.dtype);
    const reduced = reduce(a2D, outputDType, 'prod', backend);
    res = reshape({inputs: {x: reduced}, backend, attrs: {shape: outShape}});

    toDispose.push(a2D);
    toDispose.push(reduced);
  }

  if (keepDims) {
    toDispose.push(res);
    const newShape = backend_util.expandShapeToKeepDim(res.shape, origAxes);
    res = reshape({inputs: {x: res}, backend, attrs: {shape: newShape}});
  }

  toDispose.forEach(t => backend.disposeIntermediateTensorInfo(t));

  return res;
}

export const prodConfig: KernelConfig = {
  kernelName: Prod,
  backendName: 'webgl',
  kernelFunc: prod as {} as KernelFunc
};
