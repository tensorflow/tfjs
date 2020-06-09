/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

import {backend_util, registerKernel, TensorInfo, util} from '@tensorflow/tfjs-core';
import {Max, MaxAttrs, MaxInputs} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {transpose} from './Transpose';

let wasmMax: (xId: number, reduceSize: number, outId: number) => void;

function setup(backend: BackendWasm): void {
  wasmMax =
      backend.wasm.cwrap('Max', null /*void*/, ['number, number, number']);
}

function max(args: {backend: BackendWasm, inputs: {}, attrs: {}}): TensorInfo {
  const {backend, inputs, attrs} = args;
  const {reductionIndices} = attrs as MaxAttrs;
  const {x} = inputs as MaxInputs;
  const xId = backend.dataIdMap.get(x.dataId).id;

  let xShape = x.shape;
  const xRank = x.shape.length;
  const xVals = backend.typedArrayFromHeap(x);

  const origAxes = util.parseAxisParam(reductionIndices, xShape);
  let axes = origAxes;
  const permutedAxes = backend_util.getAxesPermutation(axes, xRank);
  const maxInputIsTransposed = permutedAxes != null;
  if (maxInputIsTransposed) {
    const newShape: number[] = new Array(xRank);
    for (let i = 0; i < newShape.length; i++) {
      newShape[i] = xShape[permutedAxes[i]];
    }

    axes = backend_util.getInnerMostAxes(axes.length, xRank);

    const xTransposed =
        transpose({inputs: {x}, attrs: {perm: permutedAxes}, backend});

    if (backend.dataIdMap.get(xTransposed.dataId).id !== xId) {
      // If perm is not no-op.
      const xTransposedVals = backend.typedArrayFromHeap(xTransposed);
      xVals.set(xTransposedVals, 0);
      backend.disposeData(xTransposed.dataId);
    }
    xShape = newShape;
  }

  backend_util.assertAxesAreInnerMostDims('max', axes, xRank);
  const [outShape, reduceShape] =
      backend_util.computeOutAndReduceShapes(xShape, axes);
  const reduceSize = util.sizeFromShape(reduceShape);

  const out = backend.makeOutput(outShape, x.dtype);
  if (util.sizeFromShape(xShape) === 0) {
    return out;
  }

  const outId = backend.dataIdMap.get(out.dataId).id;

  wasmMax(xId, reduceSize, outId);

  return out;
}

registerKernel(
    {kernelName: Max, backendName: 'wasm', setupFunc: setup, kernelFunc: max});
