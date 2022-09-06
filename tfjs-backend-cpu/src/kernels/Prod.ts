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

import {backend_util, DataType, KernelConfig, KernelFunc, Prod, ProdAttrs, ProdInputs, TensorInfo, TypedArray, upcastType, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';
import {transpose} from './Transpose';

export function prodImpl(
    xShape: number[], xDtype: DataType, xVals: TypedArray,
    reductionAxes: number[]):
    {outVals: TypedArray, outShape: number[], outDtype: DataType} {
  const [outShape, reduceShape] =
      backend_util.computeOutAndReduceShapes(xShape, reductionAxes);
  const outDtype = upcastType(xDtype, 'int32');
  const outVals = util.makeZerosTypedArray(
                      util.sizeFromShape(outShape), outDtype) as TypedArray;
  const reduceSize = util.sizeFromShape(reduceShape);

  for (let i = 0; i < outVals.length; ++i) {
    const offset = i * reduceSize;
    let prod = 1;
    for (let j = 0; j < reduceSize; ++j) {
      prod *= xVals[offset + j];
    }
    outVals[i] = prod;
  }

  return {outVals, outShape, outDtype};
}

export function prod(
    args: {inputs: ProdInputs, backend: MathBackendCPU, attrs: ProdAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {axis, keepDims} = attrs;

  assertNotComplex(x, 'prod');

  const xRank = x.shape.length;
  const axes = util.parseAxisParam(axis, x.shape);

  const permutation = backend_util.getAxesPermutation(axes, xRank);
  let reductionAxes = axes;
  let permutedX = x;
  const intermediateTensorInfos = [];
  if (permutation != null) {
    permutedX = transpose({inputs: {x}, backend, attrs: {perm: permutation}});
    intermediateTensorInfos.push(permutedX);
    reductionAxes = backend_util.getInnerMostAxes(reductionAxes.length, xRank);
  }

  const xVals = backend.data.get(permutedX.dataId).values as TypedArray;
  const {outVals, outShape, outDtype} =
      prodImpl(permutedX.shape, permutedX.dtype, xVals, reductionAxes);

  let resultShape = outShape;
  if (keepDims) {
    resultShape = backend_util.expandShapeToKeepDim(outShape, axes);
  }

  intermediateTensorInfos.forEach(
      t => backend.disposeIntermediateTensorInfo(t));

  return backend.makeTensorInfo(resultShape, outDtype, outVals);
}

export const prodConfig: KernelConfig = {
  kernelName: Prod,
  backendName: 'cpu',
  kernelFunc: prod as {} as KernelFunc
};
