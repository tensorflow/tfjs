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

import {Any, AnyAttrs, AnyInputs, backend_util, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {permuteAxesAndTranspose} from './kernel_utils';

let wasmAny: (xId: number, reduceSize: number, outId: number) => void;

function setup(backend: BackendWasm): void {
  wasmAny = backend.wasm.cwrap(Any, null /*void*/, ['number, number, number']);
}

function any(args: {backend: BackendWasm, inputs: AnyInputs, attrs: AnyAttrs}):
    TensorInfo {
  const {backend, inputs, attrs} = args;
  const {axis, keepDims} = attrs;
  const {x} = inputs;
  const xId = backend.dataIdMap.get(x.dataId).id;
  let inputId = xId;
  let input = x;

  const {transposed, axes, originalAxes, inputWasTransposed} =
      permuteAxesAndTranspose(x, axis, backend);

  if (inputWasTransposed) {
    const transposedId = backend.dataIdMap.get(transposed.dataId).id;
    input = transposed;
    inputId = transposedId;
  }

  const inputRank = input.shape.length;
  backend_util.assertAxesAreInnerMostDims('any', axes, inputRank);
  const [outShape, reduceShape] =
      backend_util.computeOutAndReduceShapes(input.shape, axes);
  const reduceSize = util.sizeFromShape(reduceShape);

  const out = backend.makeOutput(outShape, x.dtype);
  if (util.sizeFromShape(input.shape) !== 0) {
    const outId = backend.dataIdMap.get(out.dataId).id;
    wasmAny(inputId, reduceSize, outId);
  }

  if (inputWasTransposed) {
    // dispose of the transposed tensor.
    backend.disposeData(transposed.dataId);
  }

  if (keepDims) {
    // reshape
    const newShape = backend_util.expandShapeToKeepDim(out.shape, originalAxes);
    out.shape = newShape;
  }

  return out;
}

export const anyConfig: KernelConfig = {
  kernelName: Any,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: any as {} as KernelFunc
};
