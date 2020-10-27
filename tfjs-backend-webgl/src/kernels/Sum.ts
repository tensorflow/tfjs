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

import {backend_util, KernelConfig, KernelFunc, Sum, SumAttrs, SumInputs, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';

import {sumImpl} from './Sum_impl';
import {transposeImpl} from './Transpose_impl';

export function sum(
    args: {inputs: SumInputs, attrs: SumAttrs, backend: MathBackendWebGL}) {
  const {inputs, backend, attrs} = args;

  const {x} = inputs;
  const {axis, keepDims} = attrs;
  const webglBackend = backend;

  const reductionIndices = axis;

  const xRank = x.shape.length;

  const origAxes = util.parseAxisParam(reductionIndices, x.shape);
  let axes = origAxes;
  const permutedAxes = backend_util.getAxesPermutation(axes, xRank);
  const sumInputIsTransposed = permutedAxes != null;

  let sumInput = x;
  if (sumInputIsTransposed) {
    sumInput = transposeImpl(x, permutedAxes, webglBackend);

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

  const out = sumImpl(sumInput, reduceShape, outShape, webglBackend);
  if (sumInputIsTransposed) {
    webglBackend.disposeIntermediateTensorInfo(sumInput);
  }

  return out;
}

export const sumConfig: KernelConfig = {
  kernelName: Sum,
  backendName: 'webgl',
  kernelFunc: sum as {} as KernelFunc
};
