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

import {backend_util, sumOutType, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {reduce} from '../kernel_utils/reduce';
import {reshape} from './Reshape';

import {transposeImpl} from './Transpose_impl';

export function sumImpl(
    x: TensorInfo, axis: number|number[], keepDims: boolean,
    backend: MathBackendWebGL): TensorInfo {
  const reductionIndices = axis;

  const xRank = x.shape.length;

  const origAxes = util.parseAxisParam(reductionIndices, x.shape);
  let axes = origAxes;
  const permutedAxes = backend_util.getAxesPermutation(axes, xRank);
  const sumInputIsTransposed = permutedAxes != null;

  let sumInput = x;
  if (sumInputIsTransposed) {
    sumInput = transposeImpl(x, permutedAxes, backend);

    axes = backend_util.getInnerMostAxes(axes.length, xRank);
  }

  backend_util.assertAxesAreInnerMostDims('sum', axes, xRank);
  const [sumOutShape, reduceShape] =
      backend_util.computeOutAndReduceShapes(sumInput.shape, axes);

  let outShape = sumOutShape;
  if (keepDims) {
    // rather than reshape at the end, set the target shape here.
    outShape = backend_util.expandShapeToKeepDim(sumOutShape, origAxes);
  }

  const inSize = util.sizeFromShape(reduceShape);
  const xSize = util.sizeFromShape(x.shape);
  const batchSize = xSize / inSize;
  const reshapedInput = reshape(
      {inputs: {x: sumInput}, attrs: {shape: [batchSize, inSize]}, backend});

  const outType = sumOutType(x.dtype);

  const reduced = reduce(reshapedInput, outType, 'sum', backend);
  const out =
      reshape({inputs: {x: reduced}, attrs: {shape: outShape}, backend});

  backend.disposeIntermediateTensorInfo(reshapedInput);
  backend.disposeIntermediateTensorInfo(reduced);
  if (sumInputIsTransposed) {
    backend.disposeIntermediateTensorInfo(sumInput);
  }

  return out;
}
