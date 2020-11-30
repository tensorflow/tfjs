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

import {backend_util, KernelConfig, KernelFunc, Sum, SumAttrs, SumInputs, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';
import {zeros} from '../utils/zeros_impl';
import {cast} from './Cast';
import {identity} from './Identity';
import {reshape} from './Reshape';
import {transpose} from './Transpose';

export function sum(
    args: {inputs: SumInputs, backend: MathBackendCPU, attrs: SumAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {axis, keepDims} = attrs;

  assertNotComplex(x, 'sum');

  let $x;
  if (x.dtype === 'bool') {
    $x = cast({inputs: {x}, backend, attrs: {dtype: 'int32'}});
  } else {
    $x = identity({inputs: {x}, backend});
  }

  const xRank = $x.shape.length;
  const axes = util.parseAxisParam(axis, $x.shape);
  const permutation = backend_util.getAxesPermutation(axes, xRank);

  let reductionAxes = axes;
  let permutedX = $x;
  if (permutation != null) {
    permutedX =
        transpose({inputs: {x: $x}, backend, attrs: {perm: permutation}});
    reductionAxes = backend_util.getInnerMostAxes(reductionAxes.length, xRank);
  }

  backend_util.assertAxesAreInnerMostDims(
      'sum', reductionAxes, permutedX.shape.length);

  const [outShape, reduceShape] =
      backend_util.computeOutAndReduceShapes(permutedX.shape, reductionAxes);
  const resultDtype = backend_util.upcastType(permutedX.dtype, 'int32');
  let result = zeros(backend, outShape, resultDtype);
  const reduceSize = util.sizeFromShape(reduceShape);
  const vals = backend.data.get(result.dataId).values as TypedArray;

  const aVals = backend.data.get(permutedX.dataId).values as TypedArray;
  for (let i = 0; i < vals.length; ++i) {
    const offset = i * reduceSize;
    let sum = 0;
    for (let j = 0; j < reduceSize; ++j) {
      sum += aVals[offset + j];
    }
    vals[i] = sum;
  }

  if (keepDims) {
    const newShape = backend_util.expandShapeToKeepDim(result.shape, axes);
    const oldResult = result;
    result = reshape({inputs: {x: result}, backend, attrs: {shape: newShape}});
    backend.disposeIntermediateTensorInfo(oldResult);
  }

  backend.disposeIntermediateTensorInfo($x);

  if (permutation != null) {
    backend.disposeIntermediateTensorInfo(permutedX);
  }

  return result;
}

export const sumConfig: KernelConfig = {
  kernelName: Sum,
  backendName: 'cpu',
  kernelFunc: sum as {} as KernelFunc
};
